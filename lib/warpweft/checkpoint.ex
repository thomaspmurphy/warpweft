defmodule Warpweft.Checkpoint do
  @moduledoc """
  Run-directory persistence. A run dir is self-contained:

      runs/<name>/
        config.json      # full Warpweft.Config
        tokenizer.txt    # path to the tokenizer dir used
        latest.ckpt      # params + optimiser state + step (for resume)
        best.ckpt        # params and the validation loss they scored,
                         # at the best evaluation so far (no optimiser state)
  """

  alias Warpweft.Config

  def create_run_dir(%Config{} = config, tokenizer_dir) do
    stamp = Calendar.strftime(DateTime.utc_now(), "%Y%m%d-%H%M%S")
    dir = unique_dir(Path.join("runs", stamp), 0)
    File.mkdir_p!(dir)
    Config.save(config, dir)
    File.write!(Path.join(dir, "tokenizer.txt"), tokenizer_dir)
    dir
  end

  # The stamp has one-second resolution, so a sweep launching several short
  # runs can collide. Suffix rather than overwrite an existing run.
  defp unique_dir(base, 0), do: if(File.exists?(base), do: unique_dir(base, 2), else: base)

  defp unique_dir(base, n) do
    candidate = "#{base}-#{n}"
    if File.exists?(candidate), do: unique_dir(base, n + 1), else: candidate
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

  @doc """
  The validation loss recorded in `best.ckpt`, or `:infinity` if there is
  no best checkpoint yet.

  Resuming must read this back: starting from `:infinity` would make the
  first evaluation after a resume always look like an improvement and
  overwrite a genuinely better checkpoint.
  """
  def best_val_loss(run_dir) do
    path = Path.join(run_dir, "best.ckpt")

    if File.exists?(path) do
      case load(path) do
        %{"val_loss" => loss} -> Nx.to_number(loss)
        _ -> :infinity
      end
    else
      :infinity
    end
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
