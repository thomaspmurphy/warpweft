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
    eot = bpe.special_tokens |> Map.values() |> List.first()

    {train_ids, val_ids} = split(ids, n, eot)

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
    meta = (prefix <> ".meta.json") |> File.read!() |> JSON.decode!()
    {load_bin(prefix <> ".train.bin"), load_bin(prefix <> ".val.bin"), meta}
  end

  defp load_bin(path) do
    path
    |> File.read!()
    |> Nx.from_binary({:u, 16})
    |> Nx.as_type({:s, 32})
  end

  # Document-boundary split: cut at the last end-of-text token before the
  # 90% mark so no document straddles the train/val boundary.
  defp split(ids, n, eot) when is_integer(eot) do
    target = round(n * (1.0 - @val_fraction))

    cut =
      ids
      |> Enum.with_index()
      |> Enum.reduce(target, fn
        {^eot, i}, _acc when i <= target -> i + 1
        _, acc -> acc
      end)

    Enum.split(ids, cut)
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
