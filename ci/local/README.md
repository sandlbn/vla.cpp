# ci/local — the CI jobs, on your own machine

`ci/slurm/` runs on the cluster and takes its toolchain from `/swtools`. These
scripts run the same builds, benchmarks and MuJoCo/LIBERO evaluations on a
workstation with a stock Intel install. There's no Slurm and no batch queue: run
them in a shell.

| script | does | Slurm original |
|---|---|---|
| `build.sh sycl\|ov` | build vla.cpp for SYCL or OpenVINO | `bmg_build`, `bmg_ov` |
| `bench_sycl.sh` | vla-bench latency, f32 vs bf16 activations | `bmg_bf16_bench` |
| `bench_ov.sh` | vla-bench latency on the OpenVINO GPU plugin | `bmg_ov_quant_why_flat` (timing half) |
| `libero_sycl.sh` | LIBERO task success in MuJoCo, SYCL | `bmg_libero_bf16` |
| `libero_ov.sh` | LIBERO task success in MuJoCo, OpenVINO | `bmg_ov_libero` |

Each script prints `VERDICT: ...` and `DONE` at the end, like its Slurm original.

## What it expects

| | default | override |
|---|---|---|
| oneAPI Base Toolkit (SYCL) | `/opt/intel/oneapi/setvars.sh` | `ONEAPI_ROOT` |
| OpenVINO runtime | `/opt/intel/openvino/setupvars.sh` | `OV_ROOT` |
| Intel GPU driver | system packages (`intel-opencl-icd`, `libze1`) | — |
| model cache | `~/.cache/vla` | `VLA_CACHE` |

Install OpenVINO with `sudo scripts/install_ov.sh`. It installs into
`/opt/intel/openvino`. Without root, use `scripts/install_ov.sh --prefix ~/intel`,
then set `OV_ROOT=~/intel/openvino`.

The OpenVINO scripts never source oneAPI. Its lib directory ships its own
`libOpenCL.so.1` and TBB, and the GPU plugin talks to the device through OpenCL.
Keep the two in separate shells.

The LIBERO scripts also need:

```bash
bash eval/sim/libero/setup_libero.sh          # MuJoCo + LIBERO venv, once
sudo apt install protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev
#   ...or, without root:  bash ci/build_server_deps.sh  (-> ~/opt/vla-deps)
```

## Examples

### SYCL latency, f32 vs bf16

```bash
ci/local/build.sh sycl
ci/local/bench_sycl.sh                                   # evo1 + pi0
MODELS="vrfai/evo1-libero-gguf" REPS=50 ci/local/bench_sycl.sh
```

A checkpoint that isn't in `~/.cache/vla` yet is downloaded from the Hub on
first use.

### OpenVINO latency

```bash
ci/local/build.sh ov
ci/local/bench_ov.sh                                     # BitVLA bf16:f32 bf16:f16 q8_0:f16 q4_0:f16
ARMS="bf16:f32 bf16:f16" ci/local/bench_ov.sh            # compute precision only
ARMS= MODELS="vrfai/evo1-libero-gguf" ci/local/bench_ov.sh   # another arch, at f32 and f16
DEVICE=CPU ARMS=bf16:f32 ci/local/bench_ov.sh            # OpenVINO CPU plugin
```

An arm is `WEIGHTS:PRECISION`. `WEIGHTS` is the BitVLA checkpoint flavour
(`bf16`, `q8_0` or `q4_0`). `PRECISION` is the GPU plugin's compute precision
(`f32` or `f16`). The published BitVLA file is int2, which only the CUDA/SYCL
kernels read, so the first run transcodes it into each flavour it needs. The
transcode takes about a minute on the CPU, and the result is cached next to the
original. The int2 file itself has to be downloaded once:

```bash
huggingface-cli download vrfai/bitvla-libero-gguf --local-dir ~/.cache/vla/vrfai/bitvla-libero-gguf
```

This is the number to quote, because OpenVINO profiling is off. The per-op
profiles in `ci/slurm/bmg_ov_profile_ops.sbatch` slow the run down, so only
ratios within one profiled run are meaningful.

### MuJoCo / LIBERO, SYCL

```bash
SERVERS=ON ci/local/build.sh sycl
N_EPISODES=2 MODELS=evo1 ci/local/libero_sycl.sh         # quick look, ~10 min
ci/local/libero_sycl.sh                                  # evo1 + pi0 x f32/bf16, 10 ep/task
```

### MuJoCo / LIBERO, OpenVINO

```bash
SERVERS=ON ci/local/build.sh ov
N_EPISODES=1 ARMS=q4_0:f16 ci/local/libero_ov.sh         # smoke, ~5 min
ci/local/libero_ov.sh                                    # bf16/q8_0/q4_0 at f16, 5 ep/task
ARMS="bf16:f32 bf16:f16" ci/local/libero_ov.sh           # f32 vs f16 compute
```

Results go to `outputs/local/<script>/`. When a python with matplotlib is
available, each run also writes a plot in the same format as
`docs/img/libero_*.png`: per-task success, overall success with Wilson 95%
intervals, and client-side latency. The scripts never write to `docs/img`.

For reference, the Arc Pro B70 numbers from the cluster:

| | success | ms/step |
|---|---:|---:|
| BitVLA OV, f32 weights / f32 | 100/100 | 37.5 |
| BitVLA OV, f32 weights / f16 | 100/100 | 16.0 |
| BitVLA OV, bf16 / f16 | 49/50 | 16.6 |
| BitVLA OV, q8_0 / f16 | 49/50 | 16.2 |
| BitVLA OV, q4_0 / f16 | 50/50 | 17.1 |

## Reading LIBERO results

- **Judge task success rate, with its interval.** Ten episodes per task is ten
  coin flips: 9/10 against 8/10 is not a difference. The plot draws Wilson
  intervals so that this is visible.
- **Don't compare actions between arms.** On BitVLA, a one-ULP change at the
  input to the activation quantiser moves the output action by about 0.07, so
  two healthy arms disagree action by action.
- **Check the engagement lines.** ggml falls back to the CPU silently when the
  device is missing, and the run still completes tasks. The scripts fail if
  `backend = OPENVINO` / `using device GPU` (or `activations = BF16` for a bf16
  arm) is missing from the server log, so a passing run really ran where it
  says it did.

## Troubleshooting

- **`RENDER SMOKE TEST` fails.** MuJoCo can't render headless with EGL on this
  machine. Try `MUJOCO_GL=glfw` in a desktop session, or `MUJOCO_GL=osmesa`
  (slow, CPU only).
- **An episode aborts mid-sweep inside EGL.** robosuite leaks a GL context on
  each reset. Lower `EPISODES_PER_PROC` from its default of 10. Episodes are
  independent, so smaller chunks don't change the result.
- **OpenVINO picks the wrong GPU.** Set `DEVICE=GPU.1` for `bench_ov.sh`.
  `clinfo -l` lists the devices.
- **Don't set `GGML_OPENVINO_CACHE_DIR`.** Reloading cached blobs has produced
  silently wrong actions on the GPU plugin, so the scripts unset it.
