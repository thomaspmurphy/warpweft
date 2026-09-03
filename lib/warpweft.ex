defmodule Warpweft do
  @moduledoc """
  Warpweft — a decoder-only transformer ("GPT from scratch") built on
  Nx / EXLA, with a from-scratch byte-level BPE tokenizer.

  Typical workflow:

      mix wf.data --corpus shakespeare
      mix wf.tokenizer.train --corpus shakespeare --vocab 1024
      mix wf.train --preset shakespeare_small
      mix wf.generate --run runs/<timestamp> --prompt "ROMEO:"
  """
end
