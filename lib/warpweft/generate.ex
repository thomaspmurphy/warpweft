defmodule Warpweft.Generate do
  @moduledoc """
  Fast autoregressive sampling.

  The whole per-token computation — forward pass, temperature, top-k
  masking, Gumbel-max sampling — is one jitted function over a
  *fixed-shape* `{1, block_size}` buffer, so XLA compiles it exactly once
  and every subsequent token reuses the compiled program. Running the
  forward pass on a growing sequence instead would force a fresh
  compilation at every length.

  The context buffer is right-padded with zeros and tracked by a scalar
  `len`. The causal mask guarantees positions >= len cannot influence the
  logits at `len - 1`, so the padding is mathematically invisible.

  Sampling uses the Gumbel-max trick: `argmax(logits / temp + gumbel_noise)`
  is an exact sample from `softmax(logits / temp)`, fully batched, with no
  host round-trip and no while loop.
  """

  alias Warpweft.{Checkpoint, Config, Model}
  alias Warpweft.Model.Decode
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

  @doc """
  Generates text given params, a tokenizer, and a config.

  Uses the KV-cache path when the prompt plus the requested tokens fit in
  `block_size`, and the recomputing path otherwise (the cache cannot slide
  its window — see `Warpweft.Model.Decode`). Both paths produce identical
  output for the same seed, since the cached logits match the full forward
  pass and the sampling stream is the same. Pass `cache: false` to force
  the recomputing path.
  """
  def generate(params, %BPE{} = bpe, %Config{} = cfg, prompt, opts \\ []) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 200)
    seed = Keyword.get(opts, :seed, 1337)
    stop_at_eot = Keyword.get(opts, :stop_at_eot, true)

    validate_positive!(:max_new_tokens, max_new_tokens)
    validate_temperature!(Keyword.get(opts, :temperature, 0.8))

    prompt_ids = BPE.encode(bpe, prompt)
    eot = if stop_at_eot, do: BPE.end_of_text_id(bpe)

    fits = length(prompt_ids) + max_new_tokens <= cfg.block_size
    use_cache = Keyword.get(opts, :cache, true) and fits
    opts = Keyword.put(opts, :tokenizer, bpe)

    new_ids =
      if use_cache do
        cached_loop(params, prompt_ids, cfg, max_new_tokens, seed, eot, opts)
      else
        sample_loop(build_step(cfg, opts), params, prompt_ids, cfg, max_new_tokens, seed, eot, opts)
      end

    prompt <> BPE.decode(bpe, new_ids)
  end

  @doc "Whether a prompt of `prompt_len` plus `n` new tokens can use the cache."
  def cacheable?(%Config{} = cfg, prompt_len, n), do: prompt_len + n <= cfg.block_size

  defp validate_positive!(name, value) do
    unless is_integer(value) and value > 0 do
      raise ArgumentError, "#{name} must be a positive integer, got #{inspect(value)}"
    end
  end

  # Temperature divides the logits, so zero yields infinities and then NaN
  # once Gumbel noise is added, and argmax returns an arbitrary token. Ask
  # for greedy decoding with `top_k: 1` instead.
  defp validate_temperature!(t) do
    unless is_number(t) and t > 0 do
      raise ArgumentError,
            "temperature must be greater than 0, got #{inspect(t)}. " <>
              "For greedy decoding use top_k: 1 (or a small temperature such as 0.01)."
    end
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

  @doc """
  Builds the two jitted functions the cached path uses:
  `decode.(params, token, cache, pos) -> {logits, cache}` and
  `sample.(logits, key) -> {token, key}`.

  Sampling is kept out of the decode function so the prompt can be
  prefilled without consuming the PRNG stream, which is what makes the
  cached path produce byte-identical output to the recomputing path.
  """
  def build_cached_fns(%Config{} = cfg, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 50)

    decode = Nx.Defn.jit(fn params, token, cache, pos -> Decode.step(params, token, cache, pos, cfg) end)

    # Temperature is a runtime argument so it can be changed without
    # recompiling; top-k cannot be, since `Nx.top_k` needs `k` to shape
    # its output at trace time.
    sample =
      Nx.Defn.jit(fn logits, key, temperature ->
        sample_token(Nx.divide(logits, temperature), key, top_k)
      end)

    {decode, sample}
  end

  defp cached_loop(params, prompt_ids, cfg, max_new_tokens, seed, eot, opts) do
    {decode, sample} = build_cached_fns(cfg, opts)
    temperature = Nx.tensor(Keyword.get(opts, :temperature, 0.8), type: :f32)
    emit = emitter(opts)

    prompt_ids = Enum.take(prompt_ids, -cfg.block_size)
    prompt_ids = if prompt_ids == [], do: [0], else: prompt_ids

    # Prefill: run the prompt through the cache. No sampling here, so the
    # random stream is untouched until the first generated token.
    {logits, cache} =
      prompt_ids
      |> Enum.with_index()
      |> Enum.reduce({nil, Decode.init_cache(cfg)}, fn {id, pos}, {_logits, cache} ->
        decode.(params, Nx.tensor([[id]], type: :s32), cache, Nx.tensor(pos, type: :s32))
      end)

    state = %{
      logits: logits,
      cache: cache,
      key: Nx.Random.key(seed),
      pos: length(prompt_ids),
      out: [],
      pending: ""
    }

    Enum.reduce_while(1..max_new_tokens//1, state, fn _i, state ->
      {token_t, key} = sample.(state.logits, state.key, temperature)
      token = Nx.to_number(token_t)
      state = %{state | key: key, out: [token | state.out], pending: emit.(state.pending, token)}

      cond do
        token == eot ->
          {:halt, state}

        state.pos >= cfg.block_size ->
          {:halt, state}

        true ->
          {logits, cache} =
            decode.(
              params,
              Nx.reshape(token_t, {1, 1}) |> Nx.as_type(:s32),
              state.cache,
              Nx.tensor(state.pos, type: :s32)
            )

          {:cont, %{state | logits: logits, cache: cache, pos: state.pos + 1}}
      end
    end)
    |> then(fn state ->
      emit.(state.pending, :flush)
      Enum.reverse(state.out)
    end)
  end

  # Builds a per-token emitter for the `:on_token` callback.
  #
  # Tokens are byte-level, so a single token can end mid-codepoint: an
  # accented letter or an emoji spans several tokens' worth of bytes.
  # Writing each token's bytes out as they arrive would therefore print
  # mojibake. Trailing bytes that could still be completed are held back
  # until the next token supplies the rest.
  #
  # The callback is guaranteed to receive valid UTF-8. Bytes that can
  # never form a character (which a well-trained model does not produce,
  # but a random one does) are replaced with U+FFFD rather than passed
  # through, so a terminal can render the stream safely.
  #
  # Returns a function taking the pending bytes and either a token id or
  # `:flush`, and returning the new pending bytes.
  defp emitter(opts) do
    case Keyword.get(opts, :on_token) do
      nil ->
        fn pending, _ -> pending end

      fun ->
        bpe = Keyword.fetch!(opts, :tokenizer)

        fn
          pending, :flush ->
            # Anything still pending is a truncated character.
            if pending != "", do: fun.("�")
            ""

          pending, token ->
            {ready, keep} = split_complete_utf8(pending <> BPE.decode(bpe, [token]))
            if ready != "", do: fun.(ready)
            keep
        end
    end
  end

  # Splits into {emittable valid UTF-8, bytes that may yet be completed}.
  defp split_complete_utf8(bytes) do
    case :unicode.characters_to_binary(bytes) do
      valid when is_binary(valid) ->
        {bytes, ""}

      # Truncated: the tail could still become a character.
      {:incomplete, valid, rest} ->
        {valid, rest}

      # Malformed: this byte can never start or continue a character, so
      # substitute and carry on rather than stalling the stream forever.
      {:error, valid, <<_bad, rest::binary>>} ->
        {more, keep} = split_complete_utf8(rest)
        {valid <> "�" <> more, keep}
    end
  end

  # Temperature-scaled top-k Gumbel-max sampling over {1, vocab} logits.
  defp sample_token(logits, key, top_k) do
    v = Nx.axis_size(logits, 1)

    masked =
      case top_k do
        nil ->
          logits

        k ->
          k = min(k, v)
          {top_vals, _} = Nx.top_k(logits, k: k)
          kth = Nx.slice(top_vals, [0, k - 1], [1, 1])
          Nx.select(Nx.less(logits, kth), Nx.tensor(-1.0e9, type: Nx.type(logits)), logits)
      end

    {gumbel, key} = Nx.Random.gumbel(key, shape: {1, v})
    {masked |> Nx.add(gumbel) |> Nx.argmax(axis: -1) |> Nx.reshape({}), key}
  end

  defp sample_loop(step, params, prompt_ids, cfg, max_new_tokens, seed, eot, opts) do
    block = cfg.block_size
    emit = emitter(opts)

    # Keep at most the last block tokens of the prompt; left-align in the buffer.
    context = Enum.take(prompt_ids, -block)
    len = length(context)
    buffer = Nx.tensor([context ++ List.duplicate(0, block - len)], type: :s32)

    state = %{buffer: buffer, len: max(len, 1), key: Nx.Random.key(seed), out: [], pending: ""}

    Enum.reduce_while(1..max_new_tokens//1, state, fn _i, state ->
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

      state = %{state | key: key, out: [token | state.out], pending: emit.(state.pending, token)}

      if token == eot, do: {:halt, state}, else: {:cont, state}
    end)
    |> then(fn state ->
      emit.(state.pending, :flush)
      Enum.reverse(state.out)
    end)
  end
end
