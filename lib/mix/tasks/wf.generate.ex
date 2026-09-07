defmodule Mix.Tasks.Wf.Generate do
  @shortdoc "Generates text from a trained run"
  @moduledoc """
  Samples text from a run directory (uses the best checkpoint).

      mix wf.generate --run runs/20260903-101500 --prompt "ROMEO:" -n 300
      mix wf.generate --run runs/... --temp 1.0 --top-k 0 --seed 7

  `--top-k 0` disables top-k filtering. Without `--run`, the most recent
  run directory is used.
  """

  use Mix.Task

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [
          run: :string,
          prompt: :string,
          n: :integer,
          temp: :float,
          top_k: :integer,
          seed: :integer
        ],
        aliases: [n: :n]
      )

    Mix.Task.run("app.start")

    run_dir = Warpweft.Runs.resolve!(opts[:run])

    gen_opts =
      [max_new_tokens: opts[:n], temperature: opts[:temp], seed: opts[:seed]]
      |> Enum.reject(fn {_k, v} -> is_nil(v) end)
      |> then(fn acc ->
        case opts[:top_k] do
          nil -> acc
          0 -> [{:top_k, nil} | acc]
          k -> [{:top_k, k} | acc]
        end
      end)

    prompt = Keyword.get(opts, :prompt, "")

    {us, text} = :timer.tc(fn -> Warpweft.Generate.from_run(run_dir, prompt, gen_opts) end)

    IO.puts(text)
    IO.puts("\n--- #{run_dir}, #{Float.round(us / 1_000_000, 1)}s (incl. compilation)")
  end
end
