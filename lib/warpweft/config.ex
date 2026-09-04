defmodule Warpweft.Config do
  @moduledoc """
  Model and training hyperparameters, presets, and run-directory persistence.

  Architecture variants are plain atoms resolved at trace time, so every
  combination compiles to its own specialized XLA program:

    * `:pos`  - `:rope` or `:learned`
    * `:norm` - `:rms_norm` or `:layer_norm`
    * `:mlp`  - `:swiglu` or `:gelu`
    * `:tie_embeddings` - reuse the token embedding as the output projection
  """

  defstruct corpus: "shakespeare",
            vocab_size: 1024,
            block_size: 128,
            n_layer: 4,
            n_head: 4,
            d_model: 256,
            dropout: 0.1,
            pos: :rope,
            norm: :rms_norm,
            mlp: :swiglu,
            tie_embeddings: true,
            batch_size: 32,
            peak_lr: 1.0e-3,
            min_lr_ratio: 0.1,
            warmup_steps: 200,
            total_steps: 5000,
            weight_decay: 0.1,
            clip_norm: 1.0,
            eval_every: 250,
            eval_batches: 40,
            log_every: 25,
            seed: 1337

  @type t :: %__MODULE__{}

  @presets %{
    "shakespeare_small" => %{},

    # Deliberately identical to shakespeare_small apart from the corpus and
    # the vocabulary it forces, so the two runs isolate the effect of
    # training data volume (410K vs 5.1M tokens) on the generalization gap.
    "tinystories_small" => %{
      corpus: "tinystories",
      vocab_size: 4096
    },
    "tinystories_base" => %{
      corpus: "tinystories",
      vocab_size: 4096,
      block_size: 256,
      n_layer: 6,
      n_head: 6,
      d_model: 384,
      dropout: 0.05,
      peak_lr: 6.0e-4,
      warmup_steps: 300,
      total_steps: 20_000
    }
  }

  @atom_fields [:pos, :norm, :mlp]

  def presets, do: Map.keys(@presets)

  def preset(name) do
    case Map.fetch(@presets, name) do
      {:ok, overrides} -> struct!(__MODULE__, overrides)
      :error -> raise ArgumentError, "unknown preset #{inspect(name)}; known: #{inspect(presets())}"
    end
  end

  def head_dim(%__MODULE__{d_model: d, n_head: h}) do
    if rem(d, h) != 0, do: raise(ArgumentError, "d_model #{d} not divisible by n_head #{h}")
    div(d, h)
  end

  @doc "SwiGLU hidden size: ~8/3 * d_model rounded up to a multiple of 32."
  def swiglu_hidden(%__MODULE__{d_model: d}), do: ceil(d * 8 / 3 / 32) * 32

  def save(%__MODULE__{} = config, dir) do
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "config.json"), JSON.encode!(Map.from_struct(config)))
  end

  def load(dir) do
    fields =
      dir
      |> Path.join("config.json")
      |> File.read!()
      |> JSON.decode!()
      |> Map.new(fn {k, v} ->
        k = String.to_existing_atom(k)
        v = if k in @atom_fields, do: String.to_existing_atom(v), else: v
        {k, v}
      end)

    struct!(__MODULE__, fields)
  end
end
