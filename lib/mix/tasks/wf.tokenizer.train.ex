defmodule Mix.Tasks.Wf.Tokenizer.Train do
  @shortdoc "Trains a BPE tokenizer and pre-tokenizes the corpus"
  @moduledoc """
  Trains a byte-level BPE tokenizer on a corpus, saves it under
  `data/tokenizers/<corpus>-<vocab>/`, then encodes the corpus into
  `data/tokenized/<corpus>-<vocab>.{train,val}.bin`.

      mix wf.tokenizer.train --corpus shakespeare --vocab 1024
      mix wf.tokenizer.train --corpus tinystories --vocab 4096 [--sample-mb 5]

  `--sample-mb` trains the merges on a leading sample of the corpus
  (statistically fine, much faster) while still encoding the full corpus.
  """

  use Mix.Task

  alias Warpweft.Tokenizer.{BPE, Store}
  alias Warpweft.Data.{Corpus, Dataset}

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv, strict: [corpus: :string, vocab: :integer, sample_mb: :integer])

    corpus = Keyword.get(opts, :corpus, "shakespeare")
    vocab = Keyword.get(opts, :vocab, 1024)

    Mix.Task.run("app.start")

    text = Corpus.read!(corpus)

    train_text =
      case Keyword.get(opts, :sample_mb) do
        nil -> text
        mb -> binary_part(text, 0, min(mb * 1_048_576, byte_size(text)))
      end

    specials = Corpus.special_tokens(corpus)

    IO.puts("training BPE: vocab=#{vocab} on #{byte_size(train_text)} bytes ...")

    {us, bpe} =
      :timer.tc(fn -> BPE.train(train_text, vocab, special_tokens: specials, log_every: 100) end)

    IO.puts("trained #{length(bpe.merges)} merges in #{Float.round(us / 1_000_000, 1)}s")

    # Name everything after the vocabulary actually achieved, not the one
    # requested. BPE stops merging early when the corpus runs out of pairs,
    # and `Warpweft.Train` looks both the tokenizer and the tokenized data
    # up by the model's vocab_size, so the two must agree.
    achieved = BPE.vocab_size(bpe)

    if achieved < vocab do
      IO.puts(
        "note: corpus exhausted its pairs at #{achieved} tokens, short of the #{vocab} requested. " <>
          "Train with --vocab #{achieved} (or a larger corpus)."
      )
    end

    dir = Path.join("data/tokenizers", "#{corpus}-#{achieved}")
    Store.save(bpe, dir)
    IO.puts("saved tokenizer to #{dir}")

    IO.puts("pre-tokenizing full corpus (#{byte_size(text)} bytes) ...")
    {us, meta} = :timer.tc(fn -> Dataset.pretokenize(bpe, text, corpus) end)

    IO.puts(
      "done in #{Float.round(us / 1_000_000, 1)}s: " <>
        "#{meta["train_tokens"]} train / #{meta["val_tokens"]} val tokens " <>
        "(#{Float.round(byte_size(text) / (meta["train_tokens"] + meta["val_tokens"]), 2)} bytes/token)"
    )
  end
end
