defmodule Warpweft.Data.Corpus do
  @moduledoc """
  Downloads and caches the raw training corpora under `data/raw/`.
  """

  @raw_dir "data/raw"

  @corpora %{
    "shakespeare" => %{
      url: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
      min_bytes: 1_000_000,
      special_tokens: []
    },
    "tinystories" => %{
      url: "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
      min_bytes: 20_000_000,
      special_tokens: ["<|endoftext|>"]
    }
  }

  def names, do: Map.keys(@corpora)

  def special_tokens(name), do: fetch_spec!(name).special_tokens

  def path(name) do
    fetch_spec!(name)
    Path.join(@raw_dir, "#{name}.txt")
  end

  @doc "Downloads the corpus unless already cached. Returns the local path."
  def fetch(name) do
    spec = fetch_spec!(name)
    path = path(name)

    if File.exists?(path) do
      IO.puts("#{name}: cached at #{path} (#{File.stat!(path).size} bytes)")
    else
      File.mkdir_p!(@raw_dir)
      IO.puts("#{name}: downloading #{spec.url}")
      body =
        case Req.get(spec.url, receive_timeout: 300_000, decode_body: false) do
          {:ok, %{status: 200, body: body}} ->
            body

          {:ok, %{status: status}} ->
            raise "downloading #{name} failed: HTTP #{status} from #{spec.url}"

          {:error, reason} ->
            raise "downloading #{name} failed: #{Exception.message(reason)} (#{spec.url})"
        end

      if byte_size(body) < spec.min_bytes do
        raise "download for #{name} too small: #{byte_size(body)} bytes (expected >= #{spec.min_bytes})"
      end

      File.write!(path, body)
      IO.puts("#{name}: saved #{byte_size(body)} bytes to #{path}")
    end

    path
  end

  def read!(name), do: name |> fetch() |> File.read!()

  defp fetch_spec!(name) do
    Map.get(@corpora, name) || raise ArgumentError, "unknown corpus #{inspect(name)}; known: #{inspect(names())}"
  end
end
