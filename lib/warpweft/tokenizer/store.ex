defmodule Warpweft.Tokenizer.Store do
  @moduledoc """
  Saves and loads a trained BPE tokenizer as human-inspectable files:

    * `merges.txt`  - one merge per line (`left_id right_id`), rank = line number
    * `meta.json`   - vocab size and special tokens
    * `vocab.tsv`   - id, hex bytes, printable form (informational only,
      the vocab is fully derivable from the merges)
  """

  alias Warpweft.Tokenizer.BPE

  def save(%BPE{} = bpe, dir) do
    File.mkdir_p!(dir)

    merges =
      bpe.merges
      |> Enum.map(fn {l, r} -> "#{l} #{r}\n" end)
      |> IO.iodata_to_binary()

    File.write!(Path.join(dir, "merges.txt"), merges)

    meta = %{
      "vocab_size" => BPE.vocab_size(bpe),
      "special_tokens" => bpe.special_tokens |> Enum.sort_by(fn {_tok, id} -> id end) |> Enum.map(fn {tok, _id} -> tok end)
    }

    File.write!(Path.join(dir, "meta.json"), JSON.encode!(meta))

    vocab_tsv =
      bpe.vocab
      |> Enum.sort()
      |> Enum.map(fn {id, bytes} ->
        "#{id}\t#{Base.encode16(bytes)}\t#{printable(bytes)}\n"
      end)
      |> IO.iodata_to_binary()

    File.write!(Path.join(dir, "vocab.tsv"), vocab_tsv)
    :ok
  end

  def load(dir) do
    meta = dir |> Path.join("meta.json") |> File.read!() |> JSON.decode!()

    merges =
      dir
      |> Path.join("merges.txt")
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(fn line ->
        [l, r] = line |> String.split(" ") |> Enum.map(&String.to_integer/1)
        {l, r}
      end)

    BPE.from_merges(merges, meta["special_tokens"])
  end

  defp printable(bytes) do
    if String.valid?(bytes) and not String.contains?(bytes, ["\n", "\t", "\r"]) do
      bytes
    else
      "<bytes>"
    end
  end
end
