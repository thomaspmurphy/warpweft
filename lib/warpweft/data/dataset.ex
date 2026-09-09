defmodule Warpweft.Data.Dataset do
  @moduledoc """
  Pre-tokenizes a corpus once into flat binary files of token ids
  (`data/tokenized/<name>-<vocab>.{train,val}.bin`, little-endian u16),
  and loads them back as Nx tensors.

  Tokenizing once and training from the id files keeps the training loop
  free of any text processing: the whole corpus lives on the device as a
  single 1-D tensor.
  """

  alias Warpweft.Tokenizer.BPE

  @tokenized_dir "data/tokenized"
  @val_fraction 0.1

  def prefix(corpus_name, vocab_size),
    do: Path.join(@tokenized_dir, "#{corpus_name}-#{vocab_size}")

  @doc """
  Encodes `text` with `bpe` and writes train/val binary files.

  When the tokenizer has an end-of-text special token, the split is done
  on document boundaries; otherwise on a raw token-count boundary.
  """
  def pretokenize(bpe, text, corpus_name) do
    ids = BPE.encode(bpe, text)
    n = length(ids)

    {train_ids, val_ids} = split(ids, n, BPE.end_of_text_id(bpe))

    prefix = prefix(corpus_name, BPE.vocab_size(bpe))
    File.mkdir_p!(@tokenized_dir)
    File.write!(prefix <> ".train.bin", to_u16_binary(train_ids))
    File.write!(prefix <> ".val.bin", to_u16_binary(val_ids))

    meta = %{
      "corpus" => corpus_name,
      "vocab_size" => BPE.vocab_size(bpe),
      "train_tokens" => length(train_ids),
      "val_tokens" => length(val_ids)
    }

    File.write!(prefix <> ".meta.json", JSON.encode!(meta))
    meta
  end

  @doc "Loads `{train, val, meta}` where train/val are 1-D s32 tensors."
  def load(corpus_name, vocab_size) do
    prefix = prefix(corpus_name, vocab_size)
    meta_path = prefix <> ".meta.json"

    unless File.exists?(meta_path) do
      raise ArgumentError, """
      No tokenized data for corpus #{inspect(corpus_name)} at vocab size #{vocab_size}.
      Expected #{meta_path}.

      Prepare it with:
          mix wf.data --corpus #{corpus_name}
          mix wf.tokenizer.train --corpus #{corpus_name} --vocab #{vocab_size}
      #{available_hint(corpus_name)}\
      """
    end

    meta = meta_path |> File.read!() |> JSON.decode!()
    {load_bin(prefix <> ".train.bin"), load_bin(prefix <> ".val.bin"), meta}
  end

  defp available_hint(corpus_name) do
    case Path.wildcard(Path.join(@tokenized_dir, "#{corpus_name}-*.meta.json")) do
      [] ->
        ""

      paths ->
        sizes =
          Enum.map_join(
            paths,
            ", ",
            &(&1 |> Path.basename(".meta.json") |> String.split("-") |> List.last())
          )

        "\nAlready tokenized at vocab size(s): #{sizes}."
    end
  end

  defp load_bin(path) do
    path
    |> File.read!()
    |> Nx.from_binary({:u, 16})
    |> Nx.as_type({:s, 32})
  end

  # Document-boundary split: cut at the last end-of-text token before the
  # 90% mark so no document straddles the train/val boundary.
  #
  # Falls back to cutting at the target when the nearest boundary is far
  # from it. Without that floor, a corpus whose only end-of-text token sits
  # near the start would silently yield a split like 1% train / 99% val.
  defp split(ids, n, eot) when is_integer(eot) do
    target = round(n * (1.0 - @val_fraction))
    floor = round(target * 0.9)

    cut =
      ids
      |> Enum.with_index()
      |> Enum.reduce(0, fn
        {^eot, i}, _acc when i <= target -> i + 1
        _, acc -> acc
      end)

    Enum.split(ids, if(cut >= floor, do: cut, else: target))
  end

  defp split(ids, n, nil) do
    Enum.split(ids, round(n * (1.0 - @val_fraction)))
  end

  defp to_u16_binary(ids) do
    ids
    |> Enum.map(&<<&1::unsigned-little-16>>)
    |> IO.iodata_to_binary()
  end
end
