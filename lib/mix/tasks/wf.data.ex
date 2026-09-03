defmodule Mix.Tasks.Wf.Data do
  @shortdoc "Downloads a training corpus into data/raw/"
  @moduledoc """
  Downloads and caches a corpus.

      mix wf.data --corpus shakespeare
      mix wf.data --corpus tinystories
  """

  use Mix.Task

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} = OptionParser.parse(argv, strict: [corpus: :string])
    corpus = Keyword.get(opts, :corpus, "shakespeare")

    Mix.Task.run("app.start")
    Warpweft.Data.Corpus.fetch(corpus)
  end
end
