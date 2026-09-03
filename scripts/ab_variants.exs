# Trains every architecture-variant combination for a short run and
# prints a comparison table.
#
#     mix run scripts/ab_variants.exs [steps]
#
# Results land in runs/ like normal runs; the table reports the final
# validation loss of each combo at equal step count and parameter shapes.

steps = System.argv() |> List.first("750") |> String.to_integer()

combos =
  for pos <- [:rope, :learned],
      norm <- [:rms_norm, :layer_norm],
      mlp <- [:swiglu, :gelu] do
    %{pos: pos, norm: norm, mlp: mlp}
  end

base = Warpweft.Config.preset("shakespeare_small")

results =
  for combo <- combos do
    config = struct!(base, Map.merge(combo, %{total_steps: steps, eval_every: steps, log_every: steps}))

    IO.puts("\n=== #{combo.pos} / #{combo.norm} / #{combo.mlp} ===")
    {us, {params, _run_dir}} = :timer.tc(fn -> Warpweft.Train.run(config) end)

    {_train, val, _meta} = Warpweft.Data.Dataset.load(config.corpus, config.vocab_size)
    eval_step = Warpweft.Train.build_eval_step(config)
    val_loss = Warpweft.Train.evaluate(eval_step, params, val, config)

    Map.merge(combo, %{
      val_loss: Float.round(val_loss, 4),
      params: Warpweft.Model.param_count(params),
      seconds: Float.round(us / 1_000_000, 1)
    })
  end

IO.puts("\n\n| pos | norm | mlp | params | val loss @#{steps} | seconds |")
IO.puts("|---|---|---|---|---|---|")

results
|> Enum.sort_by(& &1.val_loss)
|> Enum.each(fn r ->
  IO.puts("| #{r.pos} | #{r.norm} | #{r.mlp} | #{r.params} | #{r.val_loss} | #{r.seconds} |")
end)
