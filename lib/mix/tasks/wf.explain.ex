defmodule Mix.Tasks.Wf.Explain do
  @shortdoc "Walks one prompt through every stage of the model, printing what happens"
  @moduledoc """
  A teaching aid. Takes a prompt and shows each stage of the pipeline with
  real shapes and real numbers, so the abstract description of a
  transformer becomes something you can point at.

      mix wf.explain --prompt "The cat sat on the"
      mix wf.explain --run runs/20260903-081542 --prompt "ROMEO:"

  Pair it with `mix wf.attention --heatmaps`, which shows what the
  attention heads inside step 3 actually learned.
  """

  use Mix.Task

  alias Warpweft.{Checkpoint, Config, Model, Runs}
  alias Warpweft.Tokenizer.{BPE, Store}

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} = OptionParser.parse(argv, strict: [run: :string, prompt: :string])

    Mix.Task.run("app.start")

    run_dir = Runs.resolve!(opts[:run])
    prompt = Keyword.get(opts, :prompt, "The cat sat on the")

    {params, %Config{} = cfg} = Checkpoint.load_run(run_dir)
    bpe = run_dir |> Checkpoint.tokenizer_dir() |> Store.load()

    IO.puts("\n#{run_dir}: #{Model.param_count(params)} parameters, #{cfg.n_layer} layers")

    ids = tokenise(bpe, cfg, prompt)
    embed(cfg, ids)
    blocks(cfg)
    logits = output(params, cfg, bpe, ids)
    sample(logits, cfg, bpe)
  end

  defp tokenise(bpe, cfg, prompt) do
    heading(1, "TOKENISATION: text becomes integers")

    chunks = BPE.chunks(prompt)
    ids = BPE.encode(bpe, prompt) |> Enum.take(cfg.block_size)

    IO.puts("  input      #{inspect(prompt)}  (#{byte_size(prompt)} bytes)")
    IO.puts("  chunks     #{inspect(chunks)}")
    IO.puts("             ^ split before BPE, so merges never cross these boundaries")
    IO.puts("  tokens     #{inspect(Enum.map(ids, &BPE.decode(bpe, [&1])))}")
    IO.puts("  ids        #{inspect(ids)}")

    IO.puts(
      "             ^ #{length(ids)} tokens from #{byte_size(prompt)} bytes " <>
        "(#{Float.round(byte_size(prompt) / max(length(ids), 1), 2)} bytes/token), " <>
        "vocabulary of #{cfg.vocab_size}"
    )

    ids
  end

  defp embed(cfg, ids) do
    heading(2, "EMBEDDING: integers become directions in space")

    IO.puts("  lookup     {1, #{length(ids)}} ids -> {1, #{length(ids)}, #{cfg.d_model}} vectors")
    IO.puts("             ^ one learned #{cfg.d_model}-dimensional vector per token")

    case cfg.pos do
      :learned ->
        IO.puts("  position   a second learned table is added, one vector per position")

      :rope ->
        IO.puts("  position   RoPE: no vectors added here. Queries and keys get rotated")
        IO.puts("             inside attention instead, by an angle set by their position,")
        IO.puts("             so their dot product depends only on the distance between them")
    end
  end

  defp blocks(cfg) do
    heading(3, "#{cfg.n_layer} TRANSFORMER BLOCKS: vectors become better vectors")

    IO.puts("  Each block adds to a residual stream rather than replacing it:")
    IO.puts("")
    IO.puts("      x = x + attention(norm(x))      <- move information between positions")
    IO.puts("      x = x + feed_forward(norm(x))   <- process it at each position")
    IO.puts("")

    IO.puts(
      "  attention  #{cfg.n_head} heads x #{Config.head_dim(cfg)} dims. Each position builds a query," <>
        " compares"
    )

    IO.puts("             it against every key, and takes a weighted average of the values.")
    IO.puts("             Causal mask: position i may only see positions 0..i.")

    hidden =
      case cfg.mlp do
        :swiglu -> Config.swiglu_hidden(cfg)
        :gelu -> 4 * cfg.d_model
      end

    IO.puts("  mlp        #{inspect(cfg.mlp)}, #{cfg.d_model} -> #{hidden} -> #{cfg.d_model}")
    IO.puts("  norm       #{inspect(cfg.norm)}, applied before each sub-layer")
    IO.puts("")
    IO.puts("  See what the heads actually learned:  mix wf.attention --heatmaps")
  end

  defp output(params, cfg, bpe, ids) do
    heading(4, "OUTPUT: vectors become one score per vocabulary entry")

    logits = Model.forward(params, Nx.tensor([ids], type: :s32), cfg)

    IO.puts(
      "  project    {1, #{length(ids)}, #{cfg.d_model}} -> {1, #{length(ids)}, #{cfg.vocab_size}}"
    )

    tied = if cfg.tie_embeddings, do: "reuses the embedding table, transposed", else: "a separate matrix"
    IO.puts("             ^ #{tied}")
    IO.puts("")
    IO.puts("  Every position predicts its own next token. That is why one forward")
    IO.puts("  pass gives #{length(ids)} training examples rather than one:")
    IO.puts("")

    for {id, i} <- Enum.with_index(ids) do
      {values, indices} = logits[[0, i]] |> Nx.top_k(k: 3)
      probs = values |> Nx.subtract(Nx.reduce_max(values)) |> Nx.exp()
      probs = Nx.divide(probs, Nx.sum(probs))

      top =
        indices
        |> Nx.to_flat_list()
        |> Enum.zip(Nx.to_flat_list(probs))
        |> Enum.map_join("  ", fn {t, p} ->
          "#{inspect(BPE.decode(bpe, [t]))} #{round(p * 100)}%"
        end)

      IO.puts("    after #{String.pad_trailing(inspect(BPE.decode(bpe, [id])), 12)} -> #{top}")
    end

    IO.puts("")
    IO.puts("             (percentages are within the top 3, not the full #{cfg.vocab_size})")

    logits
  end

  defp sample(logits, _cfg, bpe) do
    heading(5, "SAMPLING: scores become one token")

    {_b, t, v} = Nx.shape(logits)
    last = logits[[0, t - 1]] |> Nx.reshape({1, v})

    IO.puts("  Taking the most likely token every time gives flat, repetitive text,")
    IO.puts("  so we sample. Temperature sharpens or flattens the distribution;")
    IO.puts("  top-k discards the long tail first.")
    IO.puts("")

    for temperature <- [0.2, 0.8, 1.5] do
      draws =
        Enum.map(1..12, fn seed ->
          scaled = Nx.divide(last, temperature)
          {gumbel, _} = Nx.Random.gumbel(Nx.Random.key(seed), shape: {1, v})
          scaled |> Nx.add(gumbel) |> Nx.argmax() |> Nx.to_number()
        end)

      distinct = draws |> Enum.uniq() |> length()

      IO.puts(
        "    temperature #{temperature}  ->  #{distinct} distinct tokens in 12 draws: " <>
          (draws |> Enum.uniq() |> Enum.take(5) |> Enum.map_join(" ", &inspect(BPE.decode(bpe, [&1]))))
      )
    end

    IO.puts("")
    IO.puts("  Then the chosen token is appended and the whole thing runs again.")
    IO.puts("  Watch it happen:  mix wf.repl\n")
  end

  defp heading(n, text) do
    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("#{n}. #{text}")
    IO.puts(String.duplicate("=", 70))
  end
end
