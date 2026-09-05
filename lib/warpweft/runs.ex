defmodule Warpweft.Runs do
  @moduledoc """
  Locating trained runs on disk.

  A run directory is only usable if it holds both a config and at least one
  checkpoint. A run interrupted before its first evaluation has the config
  but no checkpoint, and being the newest directory it would otherwise be
  the one picked by default and then fail deep inside loading.
  """

  @runs_dir "runs"

  @doc "Usable run directories, newest first."
  def list do
    case File.ls(@runs_dir) do
      {:ok, entries} ->
        entries
        |> Enum.sort(:desc)
        |> Enum.map(&Path.join(@runs_dir, &1))
        |> Enum.filter(&usable?/1)

      {:error, _} ->
        []
    end
  end

  @doc "True if `dir` holds a config and at least one checkpoint."
  def usable?(dir) do
    File.dir?(dir) and File.exists?(Path.join(dir, "config.json")) and
      (File.exists?(Path.join(dir, "best.ckpt")) or File.exists?(Path.join(dir, "latest.ckpt")))
  end

  @doc """
  Resolves a run directory: the given one if provided, otherwise the newest
  usable one. Raises with an actionable message rather than failing later
  inside a file read.
  """
  def resolve!(nil) do
    case list() do
      [newest | _] ->
        newest

      [] ->
        raise ArgumentError, """
        No trained models found in #{@runs_dir}/.

        Train one with:
            mix wf.data --corpus shakespeare
            mix wf.tokenizer.train --corpus shakespeare --vocab 1024
            mix wf.train --preset shakespeare_small
        #{unusable_hint()}\
        """
    end
  end

  def resolve!(dir) do
    cond do
      usable?(dir) ->
        dir

      not File.dir?(dir) ->
        raise ArgumentError, "Run directory #{inspect(dir)} does not exist.#{available_hint()}"

      true ->
        raise ArgumentError,
              "#{dir} is not a usable run: it has no checkpoint " <>
                "(expected best.ckpt or latest.ckpt). It was probably interrupted " <>
                "before its first evaluation.#{available_hint()}"
    end
  end

  defp available_hint do
    case list() do
      [] -> ""
      runs -> "\n\nAvailable runs:\n  " <> Enum.join(runs, "\n  ")
    end
  end

  # Distinguishes "nothing there at all" from "directories exist but none
  # got far enough to checkpoint", which are different user problems.
  defp unusable_hint do
    case File.ls(@runs_dir) do
      {:ok, entries} when entries != [] ->
        "\nFound #{length(entries)} directory/directories under #{@runs_dir}/, " <>
          "but none contains a checkpoint."

      _ ->
        ""
    end
  end
end
