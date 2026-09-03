defmodule Warpweft.MixProject do
  use Mix.Project

  def project do
    [
      app: :warpweft,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
