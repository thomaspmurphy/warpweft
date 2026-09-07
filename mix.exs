defmodule Warpweft.MixProject do
  use Mix.Project

  def project do
    [
      app: :warpweft,
      version: "0.1.0",
      elixir: "~> 1.19",
      description:
        "A decoder-only transformer built from scratch in Elixir with Nx, " <>
          "including a byte-level BPE tokenizer, KV-cache generation and " <>
          "swappable architecture variants.",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package()
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      files: ~w(lib scripts test/fixtures mix.exs mix.lock README.md LICENSE docs .formatter.exs)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/CONCEPTS.md",
        "docs/TECHNIQUES.md",
        "docs/FINDINGS.md",
        "LICENSE"
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.13"},
      {:exla, "~> 0.13"},
      {:axon, "~> 0.8"},
      {:polaris, "~> 0.1"},
      {:req, "~> 0.5"},
      {:stream_data, "~> 1.1", only: [:dev, :test]}
    ]
  end
end
