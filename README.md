<img src="ane-training/training/dashboard.gif" width="720" alt="ANE training dashboard">

# ANEgpt

**Train GPT models directly on Apple's Neural Engine.**

ANEgpt is an open-source project that runs transformer training — forward pass, backward pass, and weight updates — on the Apple Neural Engine (ANE) in Apple Silicon. No CoreML training APIs. No Metal. No GPU. Pure ANE compute, driven through reverse-engineered private APIs.

Apple does not expose any public API for training on the ANE. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

---

## What This Does

- Constructs MIL (Model Intermediate Language) programs at runtime in Objective-C
- Compiles them in-memory to ANE programs via `_ANEInMemoryModelDescriptor` (no `.mlmodelc` on disk)
- Passes tensors via IOSurface shared memory in `[1, C, 1, S]` fp16 format
- Runs forward and backward dx passes on ANE; weight gradients (dW) on CPU via Accelerate cblas
- Includes Adam optimizer, gradient accumulation, and checkpoint/resume
- Works around the ~119 ANE compile limit per process via `exec()` restart

### What You Can Train

The main Obj-C training program (`train_large.m`) trains a **Stories110M** model — a 12-layer Llama2-architecture transformer (dim=768, hidden=2048, heads=12, seq=256, vocab=32000) on TinyStories data. This is hardcoded in `stories_config.h`.

The Python-based trainer (`ane_train.py`) trains a smaller configurable GPT model through the ANE bridge, using nanochat's infrastructure for tokenization and data loading.

---

## Architecture

### Training Loop — 6 ANE Kernel Types Per Layer

Each transformer layer uses 6 ANE kernel dispatches:

| Kernel | Function |
|--------|----------|
| `fwdAttn` | RMSNorm → QKV projection → SDPA → output projection |
| `fwdFFN` | RMSNorm → SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwd` | FFN backward (W2ᵀ → SiLU_bwd → W1ᵀ, W3ᵀ) |
| `sdpaBwd1` | Woᵀ → SDPA backward part 1 (dV, probs, dp) |
| `sdpaBwd2` | SDPA backward part 2 (softmax grad → dQ, dK) |
| `qkvBwd` | QKV backward (Wqᵀ + Wkᵀ + Wvᵀ → dx) |

For the 12-layer Stories110M model, this means **72 ANE kernels per compile** (60 weight-bearing + 12 weight-free sdpaBwd2).

CPU handles: RMSNorm backward, residual connections, cross-entropy loss, classifier matmul, dW gradient accumulation (cblas_sgemm), Adam optimizer.

### System Stack

```
┌───────────────────────────────────────────────┐
│              Python (ane_train.py)             │
│  GPT model · loss · optimizer · data pipeline  │
├───────────────────────────────────────────────┤
│            ane_bridge.py (ctypes)              │
├───────────────────────────────────────────────┤
│         libane_bridge.dylib (Obj-C)           │
│   _ANEInMemoryModelDescriptor · IOSurface     │
├───────────────────────────────────────────────┤
│          Apple Neural Engine (ANE)             │
│    MIL programs · fp16 · [1,C,1,S] layout     │
└───────────────────────────────────────────────┘
```

---

## Components

### `ane-training/` — ANE Runtime & Kernels (Obj-C)

The low-level engine. Reverse-engineers private `AppleNeuralEngine.framework` APIs to compile and run MIL programs on ANE.

- **`training/train_large.m`** — Main training program: 12-layer Stories110M, full forward/backward, checkpoint, exec() restart
- **`training/stories_*.h`** — Config, IO, MIL generators, CPU ops
- **`inmem_*.m`, `sram_*.m`** — ANE benchmarks and hardware probes
- **`bridge/`** — C-callable shared library for Python access

### `nanochat/` — LLM Training Harness (Python)

A fork of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat). Covers tokenization, pretraining, SFT, RLHF, evaluation, and a ChatGPT-like web UI. Extended here with:

- **`scripts/ane_train.py`** — ANE training backend that routes linear layers through the ANE bridge
- **`runs/runane.sh`** — Script to build the bridge and run ANE training

---

## Performance

The code measures and prints performance metrics at runtime (ms/step, TFLOPS, ANE utilization %). When you run `train_large`, the efficiency report at the end shows these computed metrics.

**What the code computes** (from `train_large.m` lines 654–667): 
- Per-step timing breakdown: ANE eval, IO (fp16 conversion), classifier (cblas), cross-entropy, RMSNorm, cblas wait
- Sustained ANE TFLOPS = ANE FLOPs executed / train time
- ANE utilization % = sustained TFLOPS / 15.8 (Apple's published M4 ANE peak)

**Reported in `training/README.md`** for the 12-layer Stories110M config (dim=768, hidden=2048, seq=256):

| Component | Time (ms/step) |
|-----------|----------------|
| ANE eval | 9.6 |
| IO (fp16 conversion) | 4.1 |
| Classifier (cblas) | 9.1 |
| Cross-entropy + residuals | 14.4 |
| RMSNorm | 0.1 |
| **Total** | **107 ms/step** |

> **Note:** These numbers come from the upstream `ane-training` project and have not been independently verified by us. Your results will vary by hardware (M1/M2/M3/M4/M5) and macOS version.

### Key Optimizations (from `ane-training`)

- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates transpose overhead
- **NEON vectorized fp16↔fp32** — ARM NEON intrinsics for fast IOSurface data transfer
- **vDSP cross-entropy** — `vvexpf` + `vDSP_sve` path, faster than scalar
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a background dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for overlap
- **Forward taps** — Q, K, V, attention scores exposed via concat outputs, avoiding CPU recompute

---

## Hardware Probing Results

The `m5result.md` file documents actual hardware probing results from an **M5** (ANE H16 family, same as M4), run on 2026-03-01:

- **Weights are baked at compile time** — overwriting weight blobs and reloading does not change output. Recompilation is required when weights change.
- **QoS has no effect on ANE frequency** — all QoS values 0-63 produce identical latency (~0.07ms avg for a 256×256 conv)
- **`_ANEPerformanceStats`** has `hwExecutionTime` property for wall-clock ANE timing, but requires `perfStatsMask` to be set before eval
- **`_ANEChainingRequest`** exists with loopback support — could enable multi-layer execution without CPU round-trips (unexplored)

---

## Getting Started

### Requirements

- **macOS 15+** on Apple Silicon (tested on M4, M5)
- **Xcode Command Line Tools** (`xcode-select --install`)
- **Python 3.10+** and [uv](https://docs.astral.sh/uv/) (for nanochat path only)

### Option A: Pure Obj-C training (ane-training)

```bash
cd ane-training

# You need pretokenized data first
cd training && python3 tokenize.py && cd ..

# Build and run
cd training && make train_large && ./train_large
```

### Option B: Python-based training (nanochat + ANE bridge)

```bash
cd nanochat
bash runs/runane.sh
```

This will set up the virtual environment, build the ANE bridge, and train a tiny model on synthetic data. See `runs/runane.sh` for details.

---

## Known Limitations

- **~119 compile limit** — ANE compiler leaks resources per process; worked around via `exec()` restart with checkpoint save/restore
- **Weights baked at compile time** — every weight update requires recompilation of all kernels (verified on M5, see `m5result.md`)
- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@Kᵀ (ANE) → mask+softmax → scores@V (ANE)
- **macOS only** — requires Apple Silicon and private framework APIs
- **Undocumented APIs** — may break with macOS updates

---

## TODO

- [ ] Multi-layer chaining via `_ANEChainingRequest` to reduce CPU round-trips between layers
- [ ] Explore `_ANEPerformanceStats.hwExecutionTime` for accurate ANE timing
- [ ] Real-time eval path (`evaluateRealTimeWithModel:`) for lower latency
- [ ] Higher accumulation steps to amortize compile cost
- [ ] Integration with nanochat's SFT/RLHF stages on ANE
- [ ] Compatibility testing across Apple Silicon generations (M1/M2/M3/M4/M5)
- [ ] Document discovered MIL instructions and ANE behavior

---

## Acknowledgements

This project builds on the following work:

- **[maderix/ane-training](https://github.com/maderix/ane-training)** — The original reverse-engineering of ANE private APIs for neural network training. The `ane-training/` directory is based on this work.
- **[Andrej Karpathy / nanochat](https://github.com/karpathy/nanochat)** — The simplest full-stack LLM training harness. The `nanochat/` directory is a fork extended with ANE training support.
- **[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** — Gamified nanoGPT with leaderboards, which inspired nanochat's speedrun approach.

---

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for **research and educational purposes** under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is **not affiliated with or endorsed by Apple Inc.** Use at your own risk.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The included components also carry MIT licenses:
- `ane-training/` — MIT © 2026 maderix
- `nanochat/` — MIT © 2025 Andrej Karpathy

---

## Cite

```bibtex
@misc{anegpt,
  author = {Vipul Divyanshu},
  title = {ANEgpt: Training GPT Models on Apple Neural Engine},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/vipuldivyanshu/ANEgpt}
}
```

---

<p align="center">
  <i>Built with curiosity, reverse engineering, and a healthy disregard for "inference only."</i>
</p>
# ANEgpt
