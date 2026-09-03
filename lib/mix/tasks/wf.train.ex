defmodule Mix.Tasks.Wf.Train do
  @shortdoc "Trains a warpweft model"
  @moduledoc """
  Trains a model and writes checkpoints to a fresh `runs/<timestamp>/` dir.

      mix wf.train --preset shakespeare_small
      mix wf.train --preset shakespeare_small --steps 1000 --pos learned --norm layer --mlp gelu --no-tie
      mix wf.train --resume runs/20260903-101500

  Variant flags: `--pos rope|learned`, `--norm rms|layer`, `--mlp swiglu|gelu`,
  `--no-tie` (untie embeddings). Scalar overrides: `--steps`, `--batch`,
  `--block`, `--lr`, `--seed`.
  """

  use Mix.Task

  alias Warpweft.Config

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [
          preset: :string,
          resume: :string,
          steps: :integer,
          batch: :integer,
          block: :integer,
          lr: :float,
          seed: :integer,
          pos: :string,
          norm: :string,
          mlp: :string,
          tie: :boolean
        ]
      )

    Mix.Task.run("app.start")

    case Keyword.get(opts, :resume) do
      nil ->
        config = build_config(opts)
        Warpweft.Train.run(config)

      run_dir ->
        config = Config.load(run_dir)
        Warpweft.Train.run(config, resume: run_dir)
    end
  end

  defp build_config(opts) do
    config = Config.preset(Keyword.get(opts, :preset, "shakespeare_small"))

    overrides =
      [
        total_steps: opts[:steps],
        batch_size: opts[:batch],
        block_size: opts[:block],
        peak_lr: opts[:lr],
        seed: opts[:seed],
        pos: variant(opts[:pos], %{"rope" => :rope, "learned" => :learned}),
        norm: variant(opts[:norm], %{"rms" => :rms_norm, "layer" => :layer_norm}),
        mlp: variant(opts[:mlp], %{"swiglu" => :swiglu, "gelu" => :gelu}),
        tie_embeddings: opts[:tie]
      ]
      |> Enum.reject(fn {_k, v} -> is_nil(v) end)

    struct!(config, overrides)
  end

  defp variant(nil, _mapping), do: nil

  defp variant(value, mapping) do
    Map.get(mapping, value) ||
      raise ArgumentError, "expected one of #{inspect(Map.keys(mapping))}, got #{inspect(value)}"
  end
end
