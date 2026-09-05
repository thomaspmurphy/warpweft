defmodule Mix.Tasks.Wf.Repl do
  @shortdoc "Interactive prompt: type text, watch the model continue it"
  @moduledoc """
  Loads a trained model once and generates from whatever you type, printing
  tokens as they arrive.

      mix wf.repl
      mix wf.repl --run runs/20260904-173641 --temp 0.9

  Inside the prompt, anything starting with `/` is a command:

      /temp 1.0      sampling temperature (higher = more random)
      /n 200         how many tokens to generate
      /top-k 50      keep only the k most likely tokens (0 disables)
      /seed 42       fix the seed, or /seed random
      /settings      show current settings
      /help
      /quit

  Everything else is treated as a prompt.
  """

  use Mix.Task

  alias Warpweft.{Checkpoint, Generate}
  alias Warpweft.Tokenizer.Store

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [run: :string, temp: :float, n: :integer, top_k: :integer, seed: :integer]
      )

    Mix.Task.run("app.start")

    run_dir = Keyword.get_lazy(opts, :run, &latest_run!/0)

    IO.puts("loading #{run_dir} ...")
    {params, cfg} = Checkpoint.load_run(run_dir)
    bpe = run_dir |> Checkpoint.tokenizer_dir() |> Store.load()

    settings = %{
      temperature: Keyword.get(opts, :temp, 0.8),
      n: Keyword.get(opts, :n, 120),
      top_k: Keyword.get(opts, :top_k, 50),
      seed: Keyword.get(opts, :seed, :random)
    }

    IO.puts(
      "#{Warpweft.Model.param_count(params)} params, #{cfg.n_layer}x#{cfg.n_head}, " <>
        "context #{cfg.block_size}, vocab #{cfg.vocab_size}, corpus #{cfg.corpus}"
    )

    # Warm up so the first real prompt is not paying for XLA compilation.
    IO.write("compiling ... ")
    Generate.generate(params, bpe, cfg, "a", max_new_tokens: 1, seed: 0)
    IO.puts("ready\n")
    IO.puts(~s(Type a prompt and press enter. "/help" for commands, "/quit" to exit.))

    loop(%{params: params, bpe: bpe, cfg: cfg, settings: settings})
  end

  defp loop(state) do
    case IO.gets("\nwarpweft> ") do
      :eof ->
        IO.puts("")

      {:error, reason} ->
        IO.puts("input error: #{inspect(reason)}")

      line ->
        case String.trim(line) do
          "" -> loop(state)
          "/quit" -> IO.puts("bye")
          "/exit" -> IO.puts("bye")
          "/" <> command -> state |> handle_command(command) |> loop()
          prompt -> state |> generate(prompt) |> loop()
        end
    end
  end

  defp generate(state, prompt) do
    %{settings: s} = state
    seed = if s.seed == :random, do: :rand.uniform(1_000_000), else: s.seed

    IO.write("\n" <> prompt)

    {us, _} =
      :timer.tc(fn ->
        Generate.generate(state.params, state.bpe, state.cfg, prompt,
          max_new_tokens: s.n,
          temperature: s.temperature,
          top_k: if(s.top_k == 0, do: nil, else: s.top_k),
          seed: seed,
          on_token: &IO.write/1
        )
      end)

    IO.puts("\n\n[#{s.n} tokens, #{Float.round(us / 1000, 0)} ms, seed #{seed}]")
    state
  end

  defp handle_command(state, command) do
    case String.split(command, " ", parts: 2) do
      ["help"] ->
        IO.puts("""
          /temp 1.0      sampling temperature (higher = more random)
          /n 200         how many tokens to generate
          /top-k 50      keep only the k most likely tokens (0 disables)
          /seed 42       fix the seed, or /seed random
          /settings      show current settings
          /quit
        Anything else is used as a prompt.\
        """)

        state

      ["settings"] ->
        IO.inspect(state.settings, label: "settings")
        state

      ["temp", value] ->
        put_number(state, :temperature, value, &(&1 > 0))

      ["n", value] ->
        put_number(state, :n, value, &(&1 > 0))

      ["top-k", value] ->
        put_number(state, :top_k, value, &(&1 >= 0))

      ["seed", "random"] ->
        IO.puts("seed: random")
        put_in(state.settings.seed, :random)

      ["seed", value] ->
        put_number(state, :seed, value, fn _ -> true end)

      other ->
        IO.puts("unknown command #{inspect(other)} — try /help")
        state
    end
  end

  defp put_number(state, key, value, valid?) do
    parsed =
      case key do
        :temperature -> Float.parse(value)
        _ -> Integer.parse(value)
      end

    case parsed do
      {number, _} ->
        if valid?.(number) do
          IO.puts("#{key}: #{number}")
          put_in(state.settings[key], number)
        else
          IO.puts("#{key}: #{number} is out of range")
          state
        end

      :error ->
        IO.puts("could not parse #{inspect(value)} as a number")
        state
    end
  end

  defp latest_run! do
    case "runs" |> File.ls!() |> Enum.sort(:desc) |> List.first() do
      nil -> raise "no runs found; train one first with: mix wf.train"
      dir -> Path.join("runs", dir)
    end
  end
end
