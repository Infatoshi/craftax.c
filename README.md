# craftax.c (retired)

This repo has moved. The pure C / AVX-512 CPU port of
[Craftax-Classic](https://github.com/MichaelTMatthews/Craftax) now lives as a
single self-contained file, `craftax.c`, inside the CUDA repo:

**https://github.com/Infatoshi/craftax.cu**

That repo carries both implementations of the same game logic:

- `craftax.cu` + `main.cu` -- the CUDA version (306M SPS on an RTX PRO 6000
  Blackwell with the policy fused into a rollout megakernel)
- `craftax.c` -- this CPU port (47.8M SPS on a Ryzen 9 9950X3D), buildable
  with a single `gcc` command, no CUDA required

The full multi-file history of this repo is preserved in its git log.
