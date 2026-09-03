defmodule Warpweft.Generate do
  @moduledoc """
  Fast autoregressive sampling.

  The whole per-token computation — forward pass, temperature, top-k
  masking, Gumbel-max sampling — is one jitted function over a
  *fixed-shape* `{1, block_size}` buffer, so XLA compiles it exactly once
  and every subsequent token reuses the compiled program. (The reference
  livebook re-ran `Axon.predict` on a growing sequence: a fresh
  compilation per shape and a full re-dispatch per token.)

  The context buffer is right-padded with zeros and tracked by a scalar
  `len`. The causal mask guarantees positions >= len cannot influence the
  logits at `len - 1`, so the padding is mathematically invisible.

  Sampling uses the Gumbel-max trick: `argmax(logits / temp + gumbel_noise)`
  is an exact sample from `softmax(logits / temp)`, fully batched, with no
  host round-trip and no while loop.
  """

  alias Warpweft.{Checkpoint, Config, Model}
  alias Warpweft.Tokenizer.{BPE, Store}

  @doc """
  Generates text from a run directory.

  Options: `:max_new_tokens` (200), `:temperature` (0.8), `:top_k` (50,
  `nil` disables), `:seed` (1337), `:stop_at_eot` (true).
  """
  def from_run(run_dir, prompt, opts \\ []) do
    {params, config} = Checkpoint.load_run(run_dir)
    bpe = run_dir |> Checkpoint.tokenizer_dir() |> Store.load()
    generate(params, bpe, config, prompt, opts)
  end

  @doc "Generates text given params, a tokenizer, and a config."
  def generate(params, %BPE{} = bpe, %Config{} = cfg, prompt, opts \\ []) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 200)
    seed = Keyword.get(opts, :seed, 1337)
    stop_at_eot = Keyword.get(opts, :stop_at_eot, true)

    prompt_ids = BPE.encode(bpe, prompt)
    eot = if stop_at_eot, do: bpe.special_tokens |> Map.values() |> List.first()

    step = build_step(cfg, opts)

    new_ids = sample_loop(step, params, prompt_ids, cfg, max_new_tokens, seed, eot)
    prompt <> BPE.decode(bpe, new_ids)
  end

  @doc """
  Builds the jitted `(params, buffer, len, key) -> {token, key}` step.
  Temperature and top-k are baked in at trace time.
  """
  def build_step(%Config{} = cfg, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 0.8)
    top_k = Keyword.get(opts, :top_k, 50)

    Nx.Defn.jit(fn params, buffer, len, key ->
      logits = Model.forward(params, buffer, cfg)
      {_, _, v} = Nx.shape(logits)

      last =
        logits
        |> Nx.slice([0, Nx.subtract(len, 1), 0], [1, 1, v])
        |> Nx.reshape({1, v})
        |> Nx.divide(temperature)

      masked =
        case top_k do
          nil ->
            last

          k ->
            {top_vals, _} = Nx.top_k(last, k: min(k, v))
            kth = Nx.slice(top_vals, [0, min(k, v) - 1], [1, 1])
            Nx.select(Nx.less(last, kth), Nx.tensor(-1.0e9, type: Nx.type(last)), last)
        end

      {gumbel, key} = Nx.Random.gumbel(key, shape: {1, Nx.axis_size(masked, 1)})
      token = masked |> Nx.add(gumbel) |> Nx.argmax(axis: -1) |> Nx.reshape({})

      {token, key}
    end)
  end

  defp sample_loop(step, params, prompt_ids, cfg, max_new_tokens, seed, eot) do
    block = cfg.block_size

    # Keep at most the last block tokens of the prompt; left-align in the buffer.
    context = Enum.take(prompt_ids, -block)
    len = length(context)
    buffer = Nx.tensor([context ++ List.duplicate(0, block - len)], type: :s32)

    state = %{buffer: buffer, len: max(len, 1), key: Nx.Random.key(seed), out: []}

    Enum.reduce_while(1..max_new_tokens, state, fn _i, state ->
      {token_t, key} = step.(params, state.buffer, Nx.tensor(state.len, type: :s32), state.key)
      token = Nx.to_number(token_t)
      token_2d = Nx.reshape(token_t, {1, 1})

      state =
        if state.len < block do
          buffer = Nx.put_slice(state.buffer, [0, state.len], token_2d)
          %{state | buffer: buffer, len: state.len + 1}
        else
          # Buffer full: slide the window one token left.
          kept = Nx.slice_along_axis(state.buffer, 1, block - 1, axis: 1)
          %{state | buffer: Nx.concatenate([kept, token_2d], axis: 1)}
        end

      state = %{state | key: key, out: [token | state.out]}

      if token == eot, do: {:halt, state}, else: {:cont, state}
    end)
    |> then(fn state -> Enum.reverse(state.out) end)
  end
end
