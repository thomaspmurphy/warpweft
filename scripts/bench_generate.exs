# Benchmarks the compile-once fixed-shape generation step against the
# naive approach: re-running the forward pass on a growing sequence,
# which forces a fresh XLA compilation at every length.
#
#     mix run scripts/bench_generate.exs [run_dir]

run_dir =
  case System.argv() do
    [dir | _] -> dir
    [] -> "runs" |> File.ls!() |> Enum.sort(:desc) |> List.first() |> then(&Path.join("runs", &1))
  end

{params, cfg} = Warpweft.Checkpoint.load_run(run_dir)
IO.puts("benchmarking #{run_dir} (block=#{cfg.block_size})\n")

# --- fast path: one jitted step, fixed shapes -------------------------------

step = Warpweft.Generate.build_step(cfg, temperature: 0.8, top_k: 50)
buffer = Nx.broadcast(Nx.tensor(0, type: :s32), {1, cfg.block_size})

# warm-up (includes the single XLA compilation)
{compile_us, _} = :timer.tc(fn -> step.(params, buffer, Nx.tensor(1, type: :s32), Nx.Random.key(0)) end)

n = 200

{fast_us, _} =
  :timer.tc(fn ->
    Enum.reduce(1..n, {buffer, Nx.Random.key(1)}, fn i, {buf, key} ->
      len = Nx.tensor(min(i, cfg.block_size), type: :s32)
      {token, key} = step.(params, buf, len, key)
      buf = Nx.put_slice(buf, [0, Nx.to_number(len) - 1], Nx.reshape(token, {1, 1}))
      {buf, key}
    end)
  end)

IO.puts("fast:  compile once #{Float.round(compile_us / 1.0e6, 2)}s, " <>
        "then #{n} tokens in #{Float.round(fast_us / 1.0e6, 2)}s " <>
        "(#{Float.round(fast_us / n / 1000, 1)} ms/token)")

# --- naive path: growing shape => recompile per length ----------------------

naive_n = 15

{naive_us, _} =
  :timer.tc(fn ->
    Enum.reduce(1..naive_n, Nx.broadcast(Nx.tensor(0, type: :s32), {1, 1}), fn _i, seq ->
      # A fresh jit per call on a new shape — what Axon.predict-per-token does.
      logits = Nx.Defn.jit(fn p, s -> Warpweft.Model.forward(p, s, cfg) end).(params, seq)
      {_, t, v} = Nx.shape(logits)
      last = logits |> Nx.slice([0, t - 1, 0], [1, 1, v]) |> Nx.reshape({1, v})
      token = last |> Nx.argmax(axis: -1) |> Nx.reshape({1, 1}) |> Nx.as_type(:s32)
      Nx.concatenate([seq, token], axis: 1)
    end)
  end)

per_token_naive = naive_us / naive_n / 1000
IO.puts("naive: #{naive_n} tokens in #{Float.round(naive_us / 1.0e6, 2)}s " <>
        "(#{Float.round(per_token_naive, 1)} ms/token, recompiles per length)")

IO.puts("\nspeedup per token: #{Float.round(per_token_naive / (fast_us / n / 1000), 1)}x")
