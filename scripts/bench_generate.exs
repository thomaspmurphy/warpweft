# Compares the three ways to run autoregressive generation:
#
#   1. KV cache            - one token's projections per step, O(context)
#   2. fixed-shape recompute - full forward over the whole buffer each step,
#                              but a constant shape so XLA compiles once
#   3. naive recompute     - full forward over a *growing* sequence, forcing
#                            a fresh XLA compilation at every length
#
#     mix run scripts/bench_generate.exs [run_dir]

run_dir =
  case System.argv() do
    [dir | _] -> dir
    [] -> "runs" |> File.ls!() |> Enum.sort(:desc) |> List.first() |> then(&Path.join("runs", &1))
  end

{params, cfg} = Warpweft.Checkpoint.load_run(run_dir)
n = 100

IO.puts("#{run_dir}: block=#{cfg.block_size}, #{Warpweft.Model.param_count(params)} params")
IO.puts("generating #{n} tokens per method\n")

# --- 1. KV cache ------------------------------------------------------------

{decode, sample} = Warpweft.Generate.build_cached_fns(cfg, temperature: 0.8, top_k: 50)
cache0 = Warpweft.Model.Decode.init_cache(cfg)
tok0 = Nx.tensor([[0]], type: :s32)

{cache_compile, _} = :timer.tc(fn -> decode.(params, tok0, cache0, Nx.tensor(0, type: :s32)) end)

{cached_us, _} =
  :timer.tc(fn ->
    Enum.reduce(0..(n - 1), {cache0, Nx.Random.key(1), tok0}, fn pos, {cache, key, token} ->
      {logits, cache} = decode.(params, token, cache, Nx.tensor(pos, type: :s32))
      {next, key} = sample.(logits, key)
      {cache, key, Nx.reshape(next, {1, 1}) |> Nx.as_type(:s32)}
    end)
  end)

# --- 2. fixed-shape recompute ----------------------------------------------

step = Warpweft.Generate.build_step(cfg, temperature: 0.8, top_k: 50)
buffer = Nx.broadcast(Nx.tensor(0, type: :s32), {1, cfg.block_size})

{fixed_compile, _} = :timer.tc(fn -> step.(params, buffer, Nx.tensor(1, type: :s32), Nx.Random.key(0)) end)

{fixed_us, _} =
  :timer.tc(fn ->
    Enum.reduce(1..n, {buffer, Nx.Random.key(1)}, fn i, {buf, key} ->
      len = min(i, cfg.block_size)
      {token, key} = step.(params, buf, Nx.tensor(len, type: :s32), key)
      {Nx.put_slice(buf, [0, len - 1], Nx.reshape(token, {1, 1})), key}
    end)
  end)

# --- 3. naive recompute (growing shape) ------------------------------------

naive_n = 15

{naive_us, _} =
  :timer.tc(fn ->
    Enum.reduce(1..naive_n, Nx.broadcast(Nx.tensor(0, type: :s32), {1, 1}), fn _i, seq ->
      logits = Nx.Defn.jit(fn p, s -> Warpweft.Model.forward(p, s, cfg) end).(params, seq)
      {_, t, v} = Nx.shape(logits)
      last = logits |> Nx.slice([0, t - 1, 0], [1, 1, v]) |> Nx.reshape({1, v})
      token = last |> Nx.argmax(axis: -1) |> Nx.reshape({1, 1}) |> Nx.as_type(:s32)
      Nx.concatenate([seq, token], axis: 1)
    end)
  end)

# --- report ----------------------------------------------------------------

ms = fn us -> Float.round(us / 1000, 1) end
per = fn us, count -> Float.round(us / count / 1000, 2) end

IO.puts("kv cache          : #{per.(cached_us, n)} ms/token  (compile #{ms.(cache_compile)} ms)")
IO.puts("fixed-shape       : #{per.(fixed_us, n)} ms/token  (compile #{ms.(fixed_compile)} ms)")
IO.puts("naive (recompiles): #{per.(naive_us, naive_n)} ms/token\n")

IO.puts("kv cache vs fixed-shape: #{Float.round(fixed_us / cached_us, 1)}x")
IO.puts("kv cache vs naive      : #{Float.round(naive_us / naive_n / (cached_us / n), 1)}x")
