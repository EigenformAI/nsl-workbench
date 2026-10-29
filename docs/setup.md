# Set up a source checkout

Use this procedure from the root of your `negative-space-learning-v2` checkout.

## Prepare the host

Use Linux x86-64, Python 3.12, `uv`, and a local Docker Engine with the Compose plugin. Run the framework and Docker daemon on the same host. Remote Docker daemons are not covered because bind mounts and bridge-gateway routing use host-local paths.

Provide an NVIDIA driver and CUDA toolchain compatible with the locked Torch and Unsloth stack. Provide Git, a C and C++ compiler, CMake, a build tool such as Make or Ninja, and development headers.

Check the prerequisites:

```bash
python3.12 --version
uv --version
docker info
docker compose version
nvidia-smi
```

On WSL, `nvidia-smi` can exist at `/usr/lib/wsl/lib/nvidia-smi` without being on `PATH`. That does not mean CUDA is unavailable. Add that directory to your shell's `PATH` when needed.

Allow approximately 16 GB for the 4B snapshot and initial f16 GGUF together. Training export temporarily needs another merged checkpoint and a trained GGUF, approximately another 16 GB. Allow extra space for Python packages, Docker images, archives, and runtime outputs.

## Install the locked dependencies

```bash
uv sync --locked --python 3.12
cp .env.example .env
```

The bootstrap helper fetches its pinned public snapshot without authentication. Leave `HF_TOKEN` empty for this workflow. Other model-download paths can use that optional credential. Optional W&B tracking is disabled in the example. Do not commit `.env` or runtime outputs.

The host uses Python 3.12. The included MS-agent Dockerfile pins MS-agent 1.6.0 and MCP 1.27.0: MCP 2.x removed an API used by this harness version. These pins do not lock the base image or every transitive dependency.

## Prepare the matching model inputs

Prepare initial inference weights and the training base from the same model revision:

```bash
uv run python scripts/prepare_materials_model.py \
	--model-id Qwen/Qwen3-4B-Instruct-2507 \
	--revision cdbee75f17c01a7cc42f958dc650907174af0554 \
	--output-root .model-cache/materials-base \
	--quantization f16
```

The helper converts on CPU using checksum-pinned llama.cpp converter source. It does not train or build a server. The expected outputs are `.model-cache/materials-base/initial.gguf`, the unchanged Transformers and tokenizer snapshot in `.model-cache/materials-base/transformers/`, and model/toolchain provenance beside them. Check the helper's reported output before starting an episode. Do not substitute a similarly named third-party GGUF without checking its tokenizer, template, and source revision.

Run one bootstrap process per output root. The helper bounds conversion to 600 seconds by default; use `--conversion-timeout-seconds` to change it, or `0` to disable that timeout. Reader validation and toolchain HTTP requests have separate fixed timeouts.

Treat preparation as complete only when both `initial.gguf` and matching `provenance.json` exist. The helper does not overwrite either file. An interrupted publication can leave one file behind. Review such an output root before retrying, or choose a fresh root and update both configuration paths and the converter setting together. Snapshot downloads can reuse their verified local download metadata; conversion starts again after a failed attempt. Concurrent bootstrap writers to the same root are not supported.

Both example configurations initially select `llama:.model-cache/materials-base/initial.gguf`. Training selects `.model-cache/materials-base/transformers`.

### Prepare a CUDA inference server

Build from the same pinned source that the helper downloaded:

```bash
cmake -S .model-cache/materials-base/toolchain/llama.cpp-825eb91a66f824242761014c0cd8aea9e9191772 \
	-B .llama/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
	-DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
	-DLLAMA_BUILD_NUMBER=8606 -DLLAMA_BUILD_COMMIT=825eb91a66f824242761014c0cd8aea9e9191772
cmake --build .llama/build --target llama-server -j 4
export PATH="$PWD/.llama/build/bin:$PATH"
llama-server --version
llama-server --list-devices
```

The compiler must support the installed CUDA toolkit. NixOS can also require explicit CUDA include/library paths and runtime library configuration. No machine-specific Nix profile is supplied. The launcher uses `llama-server` from `PATH` first, then `.llama/llama-server`; its automatic Ubuntu release download is CPU-only and is not the CUDA setup above.

The copied `.env` selects `NSL_TRAIN_GGUF_CONVERTER` from this pinned source. Keep it set for the example: initial conversion and trained export must use the same converter and its bundled `gguf-py`. Without the override, the advanced Unsloth exporter can download mutable converter sources. The direct local-converter route supports its converter's `--outtype` options; it does not add a separate llama-quantize step.

The example sets `llama.n_gpu_layers = -1` for automatic memory fitting, a 65,536-token context, q4 KV caches, and flash attention. Task prompts can exceed smaller context windows. Operators can choose `0` for CPU-only inference or a positive layer count for explicit partial offload. Training still requires the GPU stack.

## Build the local images

Build the compute image from the included Compose stack:

```bash
docker compose -f docker/materials-adsorption-compose/docker-compose.yml build
```

Both configurations include `[harness.container.build] context = "docker/ms-agent"`. The entry points build that image when needed instead of requiring a pre-existing image. After changing the harness build context, add `--rebuild-harness` to your chosen entry point to force a no-cache rebuild. That flag also runs the workflow. It is not a build-only operation.

Keep the default Compose project name and both configured container IDs. They identify `agent-service` and `compute-service`. The MS-agent harness launches a separate short-lived container.

The framework starts and populates dynamic Compose environments. The compute mount starts with empty scaffolding and receives generated label-free exports at population time. Do not copy scoring labels into that directory.

## Run smoke and recursive training

```bash
uv run python scripts/generate_training_data.py --config config/config-materials-adsorption-smoke.toml
uv run python scripts/run_train_loop.py --config config/config-materials-adsorption-train.toml
```

The smoke configuration permits one episode. If you already used the episode-only entry point, select fresh smoke output paths before using the export entry point. Repeated standalone calls do not provide train-loop resume semantics. An episode without an accepted rewrite produces no task-derived training rows.

See [Runtime and recovery](runtime.md) for paths and stop/resume commands.
