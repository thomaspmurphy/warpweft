defmodule Warpweft.Tokenizer.BPETest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Warpweft.Tokenizer.{BPE, Store}

  @corpus """
  the theme of the thesis is that the theory holds.
  the theatre was there, and they thought the same.
  """

  describe "train/3" do
    test "learns the requested number of merges" do
      bpe = BPE.train(@corpus, 260)
      assert length(bpe.merges) == 4
      assert BPE.vocab_size(bpe) == 260
    end

    test "first merge on a th-heavy corpus is 't'+'h'" do
      bpe = BPE.train(@corpus, 257)
      assert bpe.merges == [{?t, ?h}]
      assert bpe.vocab[256] == "th"
    end

    test "training is deterministic" do
      assert BPE.train(@corpus, 280).merges == BPE.train(@corpus, 280).merges
    end

    test "merged token ids compress the corpus" do
      bpe = BPE.train(@corpus, 300)
      ids = BPE.encode(bpe, @corpus)
      assert length(ids) < byte_size(@corpus)
    end
  end

  describe "encode/decode" do
    test "round-trips the training corpus" do
      bpe = BPE.train(@corpus, 300)
      assert BPE.decode(bpe, BPE.encode(bpe, @corpus)) == @corpus
    end

    test "round-trips text with characters never seen in training" do
      bpe = BPE.train(@corpus, 300)
      text = "Zürich — 42 émojis 🎉 and\ttabs\n"
      assert BPE.decode(bpe, BPE.encode(bpe, text)) == text
    end

    test "special tokens get dedicated ids and round-trip" do
      bpe = BPE.train(@corpus <> "<|endoftext|>more text", 300, special_tokens: ["<|endoftext|>"])
      eot = bpe.special_tokens["<|endoftext|>"]

      ids = BPE.encode(bpe, "the end<|endoftext|>the start")
      assert eot in ids
      assert Enum.count(ids, &(&1 == eot)) == 1
      assert BPE.decode(bpe, ids) == "the end<|endoftext|>the start"
    end

    property "decode(encode(s)) == s for arbitrary strings" do
      bpe = BPE.train(@corpus, 300)

      check all(s <- StreamData.string(:utf8, max_length: 200)) do
        assert BPE.decode(bpe, BPE.encode(bpe, s)) == s
      end
    end

    property "decode(encode(s)) == s for arbitrary binaries (byte-level guarantee)" do
      bpe = BPE.train(@corpus, 300)

      check all(s <- StreamData.binary(max_length: 200)) do
        assert BPE.decode(bpe, BPE.encode(bpe, s)) == s
      end
    end
  end

  describe "Store" do
    @tag :tmp_dir
    test "save/load round-trips the tokenizer", %{tmp_dir: dir} do
      bpe = BPE.train(@corpus, 300, special_tokens: ["<|endoftext|>"])
      :ok = Store.save(bpe, dir)
      loaded = Store.load(dir)

      assert loaded.merges == bpe.merges
      assert loaded.vocab == bpe.vocab
      assert loaded.special_tokens == bpe.special_tokens

      text = "theatre theory<|endoftext|>"
      assert BPE.encode(loaded, text) == BPE.encode(bpe, text)
    end
  end
end
