defmodule Warpweft.Data.DatasetTest do
  use ExUnit.Case, async: false

  alias Warpweft.Tokenizer.BPE
  alias Warpweft.Data.Dataset

  @moduletag :tmp_dir

  setup %{tmp_dir: dir} do
    # Dataset writes under a fixed relative dir; run each test from tmp.
    old = File.cwd!()
    File.cd!(dir)
    on_exit(fn -> File.cd!(old) end)
    :ok
  end

  test "pretokenize + load round-trips ids as tensors" do
    text = String.duplicate("hello world, this is a tiny corpus. ", 50)
    bpe = BPE.train(text, 280)

    meta = Dataset.pretokenize(bpe, text, "tiny")
    assert meta["train_tokens"] + meta["val_tokens"] == length(BPE.encode(bpe, text))

    {train, val, loaded_meta} = Dataset.load("tiny", BPE.vocab_size(bpe))
    assert loaded_meta == meta
    assert Nx.type(train) == {:s, 32}
    assert Nx.size(train) == meta["train_tokens"]
    assert Nx.size(val) == meta["val_tokens"]

    ids = BPE.encode(bpe, text)
    assert Nx.to_flat_list(train) ++ Nx.to_flat_list(val) == ids
  end

  test "with an end-of-text token, the split lands on a document boundary" do
    doc = "a story about a dog. <|endoftext|>"
    text = String.duplicate(doc, 40)
    bpe = BPE.train(text, 300, special_tokens: ["<|endoftext|>"])
    eot = bpe.special_tokens["<|endoftext|>"]

    Dataset.pretokenize(bpe, text, "docs")
    {train, _val, _meta} = Dataset.load("docs", BPE.vocab_size(bpe))

    assert train |> Nx.to_flat_list() |> List.last() == eot
  end
end
