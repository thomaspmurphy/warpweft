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

  describe "chunks/1 (pre-tokenization)" do
    # The regex is the specification; the fast paths in BPE.chunks/1 are an
    # optimization that must agree with it byte-for-byte. This test owns the
    # reference copy on purpose: changing the spec should break it.
    @spec_regex ~r/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/u

    defp reference(text), do: @spec_regex |> Regex.scan(text) |> Enum.map(&hd/1)

    defp assert_matches_spec(text) do
      assert BPE.chunks(text) == reference(text),
             "chunking diverged from the spec regex for #{inspect(text, limit: 20)}"
    end

    test "splits words, numbers, punctuation and whitespace" do
      assert BPE.chunks("First Citizen:\nBefore we go") ==
               ["First", " Citizen", ":", "\n", "Before", " we", " go"]
    end

    test "glues a single leading space, but never a longer whitespace run" do
      assert BPE.chunks("a the") == ["a", " the"]
      assert BPE.chunks("a  the") == ["a", "  ", "the"]
      assert BPE.chunks("a\nthe") == ["a", "\n", "the"]
      assert BPE.chunks("a \nthe") == ["a", " \n", "the"]
    end

    test "agrees with the spec regex on tricky ASCII edge cases" do
      for text <- [
            "",
            " ",
            "  ",
            "\n",
            "\t\t",
            "a",
            " a",
            "a ",
            "42",
            " 42",
            "abc123",
            "123abc",
            "...",
            " ...",
            "!?!",
            "don't",
            "e.g. 3.14",
            "a\v\fb",
            "ALL CAPS and mixed",
            "trailing space ",
            "\nleading newline",
            String.duplicate("word ", 100)
          ] do
        assert_matches_spec(text)
      end
    end

    test "agrees with the spec regex on non-ASCII text" do
      for text <- [
            "Zürich",
            " Zürich",
            "naïve café",
            "日本語のテキスト",
            "emoji 🎉 here",
            "mixed Zürich and plain ascii words",
            "número 42",
            "a b"
          ] do
        assert_matches_spec(text)
      end
    end

    test "agrees with the spec regex across slice boundaries" do
      # Slices are cut every 64 KB, so exercise inputs far larger than one
      # slice with non-ASCII scattered near the boundaries.
      filler = String.duplicate("the quick brown fox jumps over the lazy dog. ", 4_000)

      assert_matches_spec(filler)
      assert_matches_spec(filler <> "Zürich " <> filler)
      assert_matches_spec(String.duplicate("café ", 20_000))
    end

    # Reads a checked-in fixture rather than data/raw/, which is gitignored.
    # Guarding on File.exists? instead would make this pass on a clean clone
    # without asserting anything.
    test "agrees with the spec regex on real prose" do
      assert_matches_spec(File.read!("test/fixtures/shakespeare_sample.txt"))
    end

    test "agrees with the spec regex on Unicode whitespace" do
      # `\s` in the spec regex matches Unicode whitespace, not just ASCII,
      # because Elixir's `u` modifier enables PCRE_UCP. The ASCII fast path
      # cannot recognise those bytes, so it has to defer to the regex around
      # them. Getting this wrong split a non-breaking space followed by
      # an ASCII space into two chunks instead of one.
      # Written as escapes: these characters are invisible in source.
      for text <- [
            "a\u00A0 b",
            "\u00A0 x",
            "x\u00A0\ty",
            "a\u2028 b",
            "\u3000 z",
            "word\u00A0\u00A0word",
            "trailing\u00A0"
          ] do
        assert_matches_spec(text)
      end
    end

    property "agrees with the spec regex for arbitrary valid strings" do
      check all(s <- StreamData.string(:utf8, max_length: 300)) do
        assert_matches_spec(s)
      end
    end

    test "handles invalid UTF-8 byte-by-byte instead of raising" do
      # The spec regex itself raises on invalid UTF-8, so there is nothing
      # to differentially compare against; assert the round-trip guarantee.
      assert BPE.chunks(<<72, 105, 128, 33>>) |> IO.iodata_to_binary() == <<72, 105, 128, 33>>
      assert BPE.chunks(<<0xC3>>) |> IO.iodata_to_binary() == <<0xC3>>
    end

    property "chunks always concatenate back to the input" do
      check all(s <- StreamData.binary(max_length: 300)) do
        assert s |> BPE.chunks() |> IO.iodata_to_binary() == s
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
