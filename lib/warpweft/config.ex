defmodule Warpweft.Config do
  @moduledoc """
  Model and training hyperparameters, presets, and run-directory persistence.

  Architecture variants are plain atoms resolved at trace time, so every
  combination compiles to its own specialised XLA program:

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

    # Shakespeare sized to its own data rather than to a round number.
    # The corpus is only 1.1 MB, and BPE at vocab 1024 compresses it to
    # 410K tokens against 3.5M parameters, which memorises. Byte-level
    # tokenisation (vocab 256 means zero merges) keeps all 1.0M tokens,
    # and a smaller model brings the ratio to 1.2 tokens per parameter,
    # the same regime the TinyStories run generalised in.
    "shakespeare_char" => %{
      vocab_size: 256,
      block_size: 256,
      d_model: 128,
      n_layer: 4,
      n_head: 4,
      dropout: 0.2,
      total_steps: 4000,
      eval_every: 200
    },

    # Deliberately identical to shakespeare_small apart from the corpus and
    # the vocabulary it forces, so the two runs isolate the effect of
    # training data volume (410K vs 5.1M tokens) on the generalisation gap.
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

  @allowed_variants %{
    pos: ["rope", "learned"],
    norm: ["rms_norm", "layer_norm"],
    mlp: ["swiglu", "gelu"]
  }

  @doc """
  Struct fields whose values are atoms and so need converting back from
  strings on load. Asserted against the struct in the test suite, because
  adding an atom-valued field and forgetting it here would make `load/1`
  silently return a string that only explodes later inside the model.
  """
  def atom_fields, do: @atom_fields

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

  @doc """
  Checks a config is self-consistent, raising with a specific message.

  Called from `Warpweft.Model.init/2` so problems surface immediately
  rather than as a reshape error inside attention, or (worse) silently:
  RoPE splits each head in half, so an odd `head_dim` would quietly drop a
  channel from every query and key.
  """
  def validate!(%__MODULE__{} = cfg) do
    positive = [
      vocab_size: cfg.vocab_size,
      block_size: cfg.block_size,
      n_layer: cfg.n_layer,
      n_head: cfg.n_head,
      d_model: cfg.d_model
    ]

    for {name, value} <- positive, not (is_integer(value) and value > 0) do
      raise ArgumentError, "#{name} must be a positive integer, got #{inspect(value)}"
    end

    head_dim = head_dim(cfg)

    if cfg.pos == :rope and rem(head_dim, 2) != 0 do
      raise ArgumentError,
            "RoPE needs an even head_dim, got #{head_dim} " <>
              "(d_model #{cfg.d_model} / n_head #{cfg.n_head}). " <>
              "Adjust d_model or n_head, or use pos: :learned."
    end

    unless cfg.dropout >= 0.0 and cfg.dropout < 1.0 do
      raise ArgumentError, "dropout must be in [0.0, 1.0), got #{inspect(cfg.dropout)}"
    end

    cfg
  end

  @doc "SwiGLU hidden size: ~8/3 * d_model rounded up to a multiple of 32."
  def swiglu_hidden(%__MODULE__{d_model: d}), do: ceil(d * 8 / 3 / 32) * 32

  def save(%__MODULE__{} = config, dir) do
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "config.json"), JSON.encode!(Map.from_struct(config)))
  end

  def load(dir) do
    path = Path.join(dir, "config.json")
    known = __MODULE__ |> struct() |> Map.from_struct() |> Map.keys() |> MapSet.new()

    fields =
      path
      |> File.read!()
      |> JSON.decode!()
      |> Map.new(fn {k, v} -> {parse_key!(k, known, path), v} end)
      |> Map.new(fn {k, v} -> {k, parse_value!(k, v, path)} end)

    struct!(__MODULE__, fields)
  end

  defp parse_key!(key, known, path) do
    atom = String.to_existing_atom(key)
    if MapSet.member?(known, atom), do: atom, else: unknown!(key, known, path)
  rescue
    ArgumentError -> unknown!(key, known, path)
  end

  defp unknown!(key, known, path) do
    raise ArgumentError,
          "#{path} has unknown field #{inspect(key)}. " <>
            "Known fields: #{known |> Enum.sort() |> Enum.join(", ")}."
  end

  # Variant fields are stored as strings in JSON and must come back as the
  # atoms the model dispatches on, so an unrecognised value has to fail
  # here rather than reach the forward pass as a string.
  defp parse_value!(key, value, path) when is_binary(value) do
    if key in @atom_fields do
      allowed = @allowed_variants[key]

      if value in allowed do
        String.to_existing_atom(value)
      else
        raise ArgumentError,
              "#{path} has #{key}: #{inspect(value)}, expected one of #{Enum.join(allowed, ", ")}."
      end
    else
      value
    end
  end

  defp parse_value!(_key, value, _path), do: value
end
