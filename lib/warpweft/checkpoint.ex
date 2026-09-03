defmodule Warpweft.Checkpoint do
  @moduledoc """
  Run-directory persistence. A run dir is self-contained:

      runs/<name>/
        config.json      # full Warpweft.Config
        tokenizer.txt    # path to the tokenizer dir used
        latest.ckpt      # params + optimizer state + step (for resume)
        best.ckpt        # params only, at the best validation loss
  """

  alias Warpweft.Config

  def create_run_dir(%Config{} = config, tokenizer_dir) do
    stamp = Calendar.strftime(DateTime.utc_now(), "%Y%m%d-%H%M%S")
    dir = Path.join("runs", stamp)
    File.mkdir_p!(dir)
    Config.save(config, dir)
    File.write!(Path.join(dir, "tokenizer.txt"), tokenizer_dir)
    dir
  end

  def tokenizer_dir(run_dir), do: run_dir |> Path.join("tokenizer.txt") |> File.read!() |> String.trim()

  def save_latest(run_dir, params, opt_state, step) do
    # Nx.serialize handles containers of tensors, so scalars ride along as 0-d tensors.
    write_serialized(Path.join(run_dir, "latest.ckpt"), %{
      "params" => params,
      "opt_state" => opt_state,
      "step" => Nx.tensor(step, type: :s64)
    })
  end

  def save_best(run_dir, params, val_loss) do
    write_serialized(Path.join(run_dir, "best.ckpt"), %{
      "params" => params,
      "val_loss" => Nx.tensor(val_loss, type: :f32)
    })
  end

  def load(path) do
    path |> File.read!() |> Nx.deserialize()
  end

  @doc "Loads the best (fallback: latest) params from a run dir, plus its config."
  def load_run(run_dir) do
    config = Config.load(run_dir)

    ckpt =
      cond do
        File.exists?(Path.join(run_dir, "best.ckpt")) -> load(Path.join(run_dir, "best.ckpt"))
        File.exists?(Path.join(run_dir, "latest.ckpt")) -> load(Path.join(run_dir, "latest.ckpt"))
        true -> raise "no checkpoint found in #{run_dir}"
      end

    {ckpt["params"], config}
  end

  defp write_serialized(path, container) do
    tmp = path <> ".tmp"
    File.write!(tmp, Nx.serialize(container))
    File.rename!(tmp, path)
  end
end
