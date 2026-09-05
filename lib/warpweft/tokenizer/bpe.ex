defmodule Warpweft.Tokenizer.BPE do
  @moduledoc """
  Byte-level Byte Pair Encoding, from scratch.

  The base alphabet is the 256 possible bytes, so any binary round-trips
  exactly (`decode(encode(text)) == text`) and there is never an unknown
  token. Training learns `vocab_size - 256 - length(special_tokens)`
  merges on top of the byte alphabet.

  Training uses the word-frequency formulation (Sennrich et al.): the
  corpus is pre-split into chunks with a GPT-2-style regex, merges are
  learned over the *unique chunk* frequency table rather than the raw
  corpus, which keeps pure-Elixir training tractable.
  """

  defstruct merges: [], ranks: %{}, vocab: %{}, special_tokens: %{}, special_ids: %{}

  @type t :: %__MODULE__{}

  @byte_alphabet_size 256

  # Pre-split specification: leading-space words, numbers, punctuation
  # runs, whitespace. See `chunks/1` for how it is applied.
  @chunk_regex ~r/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/u

  # Slices are small enough that their chunk lists die young rather than
  # being promoted, and large enough to amortise the per-slice overhead.
  @slice_target 65_536

  @doc """
  Trains a tokenizer on `text` up to `vocab_size` total tokens.

  `vocab_size` is an upper bound, not a guarantee: merging stops early if
  the corpus runs out of adjacent pairs, which happens with small or very
  repetitive text. Always size a model from `vocab_size/1` on the trained
  tokenizer rather than from the number you asked for.

  Options:
    * `:special_tokens` - strings assigned dedicated ids after the merges
      (e.g. `["<|endoftext|>"]`). They are split out before byte encoding
      and never participate in merges.
    * `:log_every` - print progress every N merges (nil disables).
  """
  def train(text, vocab_size, opts \\ []) do
    special_tokens = Keyword.get(opts, :special_tokens, [])
    log_every = Keyword.get(opts, :log_every)
    n_merges = vocab_size - @byte_alphabet_size - length(special_tokens)

    if n_merges < 0 do
      raise ArgumentError, "vocab_size must be at least #{@byte_alphabet_size + length(special_tokens)}"
    end

    chunk_freqs =
      text
      |> strip_specials(special_tokens)
      |> Enum.reduce(%{}, fn part, counts ->
        reduce_chunks(part, counts, fn chunk, counts ->
          Map.update(counts, :binary.bin_to_list(chunk), 1, &(&1 + 1))
        end)
      end)
      |> Map.to_list()

    merges = learn_merges(chunk_freqs, n_merges, log_every)
    from_merges(merges, special_tokens)
  end

  @doc "Rebuilds the full tokenizer struct from an ordered merge list."
  def from_merges(merges, special_tokens \\ []) do
    base_vocab = Map.new(0..(@byte_alphabet_size - 1), fn b -> {b, <<b>>} end)

    {vocab, ranks} =
      merges
      |> Enum.with_index()
      |> Enum.reduce({base_vocab, %{}}, fn {{l, r}, rank}, {vocab, ranks} ->
        new_id = @byte_alphabet_size + rank
        {Map.put(vocab, new_id, vocab[l] <> vocab[r]), Map.put(ranks, {l, r}, {rank, new_id})}
      end)

    next_id = @byte_alphabet_size + length(merges)

    {special_ids, vocab} =
      special_tokens
      |> Enum.with_index(next_id)
      |> Enum.reduce({%{}, vocab}, fn {tok, id}, {specials, vocab} ->
        {Map.put(specials, tok, id), Map.put(vocab, id, tok)}
      end)

    %__MODULE__{
      merges: merges,
      ranks: ranks,
      vocab: vocab,
      special_tokens: special_ids,
      special_ids: Map.new(special_ids, fn {tok, id} -> {id, tok} end)
    }
  end

  @doc "Total number of tokens (bytes + merges + specials)."
  def vocab_size(%__MODULE__{vocab: vocab}), do: map_size(vocab)

  @end_of_text "<|endoftext|>"

  @doc """
  The id of the end-of-text token, or `nil` if this tokenizer has none.

  Looked up by name rather than by taking the first special token, because
  map values come back in term order: with more than one special token,
  "the first one" is whichever name happens to sort first.
  """
  def end_of_text_id(%__MODULE__{special_tokens: specials}), do: Map.get(specials, @end_of_text)

  @doc """
  Encodes `text` into a list of token ids.

  Special tokens present in the text are emitted as their dedicated ids.
  Repeated chunks are memoised, so encoding a full corpus costs roughly
  one merge-application per *unique* chunk.
  """
  def encode(%__MODULE__{} = bpe, text) do
    {ids, _memo} = encode_with_memo(bpe, text, %{})
    ids
  end

  defp encode_with_memo(%__MODULE__{} = bpe, text, memo) do
    # acc is a reversed list of segments; each segment is a list of ids in order.
    {acc, memo} =
      text
      |> split_on_specials(Map.keys(bpe.special_tokens))
      |> Enum.reduce({[], memo}, fn
        {:special, tok}, {acc, memo} ->
          {[[bpe.special_tokens[tok]] | acc], memo}

        {:text, part}, {acc, memo} ->
          reduce_chunks(part, {acc, memo}, fn chunk, {acc, memo} ->
            case memo do
              %{^chunk => ids} ->
                {[ids | acc], memo}

              _ ->
                ids = chunk |> :binary.bin_to_list() |> apply_merges(bpe.ranks)
                {[ids | acc], Map.put(memo, chunk, ids)}
            end
          end)
      end)

    {acc |> Enum.reverse() |> List.flatten(), memo}
  end

  @doc "Decodes a list of token ids back into a binary."
  def decode(%__MODULE__{vocab: vocab}, ids) do
    ids
    |> Enum.map(fn id ->
      case vocab do
        %{^id => bytes} ->
          bytes

        _ ->
          raise ArgumentError,
                "token id #{id} is outside this tokenizer's vocabulary of #{map_size(vocab)}. " <>
                  "A model configured with a larger vocab_size than its tokenizer will emit " <>
                  "ids like this one; note that BPE.train/3 treats vocab_size as an upper " <>
                  "bound and stops early when the corpus runs out of pairs to merge."
      end
    end)
    |> IO.iodata_to_binary()
  end

  # -- chunking ----------------------------------------------------------------

  @doc """
  Splits a binary into pre-tokenization work units.

  This is the step before BPE proper: text is cut into words, numbers,
  punctuation runs and whitespace runs, and merges are only ever learned
  or applied *within* a unit. `@chunk_regex` is the specification of that
  split; everything below is an optimisation that must agree with it
  byte-for-byte (see the differential test in `bpe_test.exs`).

  Three things make this fast:

    * the input is cut into slices at safe boundaries, so the
      intermediate lists stay small and short-lived rather than
      materialising every chunk in the corpus at once
    * pure-ASCII slices (the vast majority of real text) are scanned by
      binary pattern matching, ~7x faster than the regex
    * only slices containing non-ASCII codepoints pay for the regex,
      which keeps the Unicode letter/number categories exact

  Invalid UTF-8 is handled byte-by-byte, so any binary round-trips.
  """
  def chunks(text) do
    text |> reduce_chunks([], fn chunk, acc -> [chunk | acc] end) |> :lists.reverse()
  end

  # Streams chunks slice by slice. Nothing ever holds more than one
  # slice's worth of chunks.
  defp reduce_chunks(text, acc, fun) do
    text
    |> slices()
    |> Enum.reduce(acc, fn slice, acc ->
      slice |> chunks_of_slice() |> Enum.reduce(acc, fun)
    end)
  end

  defp chunks_of_slice(slice), do: scan(slice, 0, [])

  defp regex_chunks(slice) do
    if String.valid?(slice) do
      @chunk_regex |> Regex.scan(slice) |> Enum.map(&hd/1)
    else
      chunks_with_invalid_bytes(slice)
    end
  end

  defp chunks_with_invalid_bytes(<<>>), do: []

  defp chunks_with_invalid_bytes(bin) do
    case :unicode.characters_to_binary(bin) do
      valid when is_binary(valid) ->
        chunks_of_slice(valid)

      {kind, valid, rest} when kind in [:error, :incomplete] ->
        <<bad_byte, rest::binary>> = rest
        chunks_of_slice(valid) ++ [<<bad_byte>>] ++ chunks_with_invalid_bytes(rest)
    end
  end

  # -- slicing -----------------------------------------------------------------

  defp slices(text) when byte_size(text) <= @slice_target, do: [text]
  defp slices(text), do: slices(text, 0, [])

  defp slices(text, start, acc) do
    size = byte_size(text)
    rest = size - start

    if rest <= @slice_target do
      :lists.reverse([:binary.part(text, start, rest) | acc])
    else
      case safe_split(text, start + @slice_target, size) do
        nil ->
          :lists.reverse([:binary.part(text, start, rest) | acc])

        stop ->
          slices(text, stop, [:binary.part(text, start, stop - start) | acc])
      end
    end
  end

  # A chunk can only span a position when both sides share a character
  # class, or when a single literal space is glued to the run after it.
  # So a whitespace byte following a non-whitespace byte is a boundary no
  # chunk can straddle.
  #
  # Both sides must be ASCII to know that. `\s` in the spec regex matches
  # *Unicode* whitespace (Elixir's `u` modifier enables PCRE_UCP), so a
  # byte >= 128 might be whitespace we cannot recognise here, and a `\s+`
  # chunk could then straddle the split. Requiring the preceding byte to
  # be ASCII and non-whitespace rules that out.
  defp safe_split(bin, from, size) when from < size do
    here = :binary.at(bin, from)
    prev = :binary.at(bin, from - 1)

    if ws?(here) and prev < 128 and not ws?(prev) do
      from
    else
      safe_split(bin, from + 1, size)
    end
  end

  defp safe_split(_bin, _from, _size), do: nil

  # -- ascii fast path ---------------------------------------------------------

  # Scans a slice with binary pattern matching, which handles ASCII only.
  # Real text is >99% ASCII but non-ASCII tends to be sprinkled throughout
  # it, so a run containing any byte >= 128 escapes to the regex on its own
  # (just far enough to reach the next boundary no chunk can straddle)
  # and then the fast path resumes. Escaping the whole slice instead would
  # mean a single curly quote taxes the other 64 KB around it.
  #
  # `i` is always at a chunk boundary, which is what makes handing an
  # isolated span to the regex give the same answer as the regex would give
  # for that span in context.
  defp scan(bin, i, acc) when i >= byte_size(bin), do: :lists.reverse(acc)

  defp scan(bin, i, acc) do
    size = byte_size(bin)
    first = :binary.at(bin, i)

    if first >= 128 do
      escape(bin, i, size, acc)
    else
      next = if i + 1 < size, do: :binary.at(bin, i + 1)

      # The regex's " ?" prefix: one literal space glued to the following
      # run. It only applies to 0x20 (not \n or \t) followed by a
      # non-whitespace character, because otherwise "\s+" claims the whole
      # whitespace run first.
      {run_start, cls} =
        if first == 0x20 and is_integer(next) and next < 128 and not ws?(next) do
          {i + 1, class(next)}
        else
          {i, class(first)}
        end

      case run_end(bin, run_start, cls, size) do
        :non_ascii -> escape(bin, i, size, acc)
        stop -> scan(bin, stop, [:binary.part(bin, i, stop - i) | acc])
      end
    end
  end

  defp escape(bin, i, size, acc) do
    stop =
      case safe_split(bin, i + 1, size) do
        nil -> size
        boundary -> boundary
      end

    chunks = bin |> :binary.part(i, stop - i) |> regex_chunks()
    scan(bin, stop, Enum.reverse(chunks, acc))
  end

  defp run_end(bin, i, cls, size) when i < size do
    c = :binary.at(bin, i)

    cond do
      c >= 128 -> :non_ascii
      class(c) == cls -> run_end(bin, i + 1, cls, size)
      true -> i
    end
  end

  defp run_end(_bin, i, _cls, _size), do: i

  # ASCII whitespace. The spec regex's `\s` is wider than this (it matches
  # Unicode whitespace too), which is why every caller here also requires
  # the byte to be ASCII before trusting the answer.
  defp ws?(c), do: c in [0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D]

  defp class(c) when c in ?a..?z or c in ?A..?Z, do: :letter
  defp class(c) when c in ?0..?9, do: :digit
  defp class(c) when c in [0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D], do: :space
  defp class(c) when c < 128, do: :other

  # -- training internals ----------------------------------------------------

  defp learn_merges(chunk_freqs, n_merges, log_every) do
    Enum.reduce_while(0..(n_merges - 1)//1, {chunk_freqs, []}, fn rank, {chunks, merges} ->
      case best_pair(chunks) do
        nil ->
          {:halt, {chunks, merges}}

        {pair, count} ->
          if log_every && rem(rank, log_every) == 0 do
            IO.puts("merge #{rank}/#{n_merges}: #{inspect(pair)} (count #{count})")
          end

          new_id = @byte_alphabet_size + rank
          chunks = Enum.map(chunks, fn {ids, freq} -> {merge_pair(ids, pair, new_id), freq} end)
          {:cont, {chunks, [pair | merges]}}
      end
    end)
    |> then(fn {_chunks, merges} -> Enum.reverse(merges) end)
  end

  defp best_pair(chunks) do
    chunks
    |> Enum.reduce(%{}, fn {ids, freq}, counts -> count_pairs(ids, freq, counts) end)
    |> Enum.max_by(fn {pair, count} -> {count, negate_pair(pair)} end, fn -> nil end)
  end

  # Deterministic tie-break: highest count, then lexicographically smallest pair.
  defp negate_pair({l, r}), do: {-l, -r}

  defp count_pairs([a, b | rest], freq, counts) do
    count_pairs([b | rest], freq, Map.update(counts, {a, b}, freq, &(&1 + freq)))
  end

  defp count_pairs(_short, _freq, counts), do: counts

  defp merge_pair([l, r | rest], {l, r} = pair, new_id), do: [new_id | merge_pair(rest, pair, new_id)]
  defp merge_pair([x | rest], pair, new_id), do: [x | merge_pair(rest, pair, new_id)]
  defp merge_pair([], _pair, _new_id), do: []

  # -- encoding internals ----------------------------------------------------

  # Repeatedly applies the lowest-rank merge present until none apply.
  defp apply_merges(ids, ranks) when map_size(ranks) == 0, do: ids

  defp apply_merges(ids, ranks) do
    case lowest_rank_pair(ids, ranks) do
      nil ->
        ids

      {pair, {_rank, new_id}} ->
        ids |> merge_pair(pair, new_id) |> apply_merges(ranks)
    end
  end

  defp lowest_rank_pair(ids, ranks) do
    ids
    |> Enum.zip(tl(ids ++ [nil]))
    |> Enum.flat_map(fn pair ->
      case ranks do
        %{^pair => rank_info} -> [{pair, rank_info}]
        _ -> []
      end
    end)
    |> Enum.min_by(fn {_pair, {rank, _id}} -> rank end, fn -> nil end)
  end

  # -- special-token splitting ----------------------------------------------

  defp strip_specials(text, []), do: [text]
  defp strip_specials(text, specials), do: String.split(text, specials)

  defp split_on_specials(text, []), do: [{:text, text}]

  defp split_on_specials(text, specials) do
    pattern = specials |> Enum.map(&Regex.escape/1) |> Enum.join("|")
    regex = Regex.compile!("(#{pattern})")

    text
    |> String.split(regex, include_captures: true)
    |> Enum.reject(&(&1 == ""))
    |> Enum.map(fn part ->
      if part in specials, do: {:special, part}, else: {:text, part}
    end)
  end
end
