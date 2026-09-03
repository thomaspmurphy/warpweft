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

  # GPT-2-ish pre-split: leading-space words, numbers, punctuation runs, whitespace.
  @chunk_regex ~r/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/u

  @doc """
  Trains a tokenizer on `text` up to `vocab_size` total tokens.

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
      |> Enum.flat_map(&chunks/1)
      |> Enum.frequencies_by(&:binary.bin_to_list/1)
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

  @doc """
  Encodes `text` into a list of token ids.

  Special tokens present in the text are emitted as their dedicated ids.
  Repeated chunks are memoized, so encoding a full corpus costs roughly
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
          part
          |> chunks()
          |> Enum.reduce({acc, memo}, fn chunk, {acc, memo} ->
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
    ids |> Enum.map(&Map.fetch!(vocab, &1)) |> IO.iodata_to_binary()
  end

  # -- chunking ----------------------------------------------------------------

  # Splits a binary into BPE work units. Valid UTF-8 goes through the
  # GPT-2-style regex; invalid bytes each become their own single-byte chunk
  # (preserving the guarantee that any binary round-trips).
  defp chunks(text) do
    if String.valid?(text) do
      @chunk_regex |> Regex.scan(text) |> Enum.map(&hd/1)
    else
      chunks_with_invalid_bytes(text)
    end
  end

  defp chunks_with_invalid_bytes(<<>>), do: []

  defp chunks_with_invalid_bytes(bin) do
    case :unicode.characters_to_binary(bin) do
      valid when is_binary(valid) ->
        chunks(valid)

      {kind, valid, rest} when kind in [:error, :incomplete] ->
        <<bad_byte, rest::binary>> = rest
        chunks(valid) ++ [<<bad_byte>>] ++ chunks_with_invalid_bytes(rest)
    end
  end

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
