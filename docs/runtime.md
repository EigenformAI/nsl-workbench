# Runtime and recovery

This reference describes the example configuration and source behavior.

## Containers and listeners

The framework host and Docker daemon are co-located. Both Compose container IDs are required: `materials-adsorption-compose-agent-service-1` and `materials-adsorption-compose-compute-service-1`. The `agent-service` container is the task environment, not the separately launched MS-agent harness.

The listeners have distinct bindings and authentication:

| Service | Binding | Authentication |
| --- | --- | --- |
| Managed llama model server | `localhost:8000` by default | The launcher adds no server API-key flag. Its client key is not server authentication. |
| OpenAI shim | `0.0.0.0`, ephemeral port | Chat completions require a per-episode bearer token. Default API documentation and schema routes are outside that route-level check. |
| MCP capability bridge | `0.0.0.0`, separate ephemeral port | HTTP middleware checks the episode bearer token before dispatch. |
| Locally launched vLLM and SGLang | All host interfaces, Docker host networking | These launch paths add no server authentication. These are advanced alternatives. |

MS-agent uses Docker bridge networking and contacts the host bridges through the discovered bridge gateway. The host shim contacts llama. MS-agent does not need direct llama access. Binding both host bridges only to loopback would break this route.

Wildcard binding includes host Docker, LAN, VPN, and public IPv4 interfaces. It does not create router forwarding or bypass NAT or firewalls. Reachable unauthenticated model servers can accept inference work and consume GPU resources. Model-management endpoints may also be reachable. The bridge token does not protect a separate model server.

An absent Docker port publication does not isolate a host-networked service. Client API keys and `NSL_*_NETWORK_MODE` readiness-address settings do not add server authentication or change listener bindings. Firewall rules, tunnels, and authenticated proxies are operator choices, not framework-enforced restrictions.

## Compute boundary

The compute service has `network_mode: none`, a read-only root filesystem, a non-root user, dropped capabilities, and `no-new-privileges`. Its read-only `/data` bind mount contains generated label-free exports only. The framework executes snippets through Docker exec. Compute receives neither labels nor the host Docker socket.

MS-agent has network access and a writable `/work` scratch mount. It is not the compute sandbox. Do not add repository-root, dataset, calibration, model-cache, or Docker-socket mounts to either container.

## Output source index

- [Smoke paths](../config/config-materials-adsorption-smoke.toml) and [training paths](../config/config-materials-adsorption-train.toml)
- [Run and trained-model paths](../scripts/run_train_loop.py)
- [Generation publications](../src/execution/generation_persistence.py) and [training artifacts](../src/train/artifacts.py)

## Credentials and diagnostic recording

Logging intentionally remains verbatim. There is no automatic credential redaction.

Known sensitive locations include:

- Episode scratch `agent.yaml`: `openai_api_key` contains the episode bridge token.
- Episode scratch `mcp_config.json`: the `Authorization` header contains that token.
- Per-step workflow YAML files in the same scratch directory: inference and MCP configuration also contain the token.
- Advanced vLLM and SGLang Docker launch-command logs: supplied HF tokens and `extra_env` values can appear verbatim in the active Loguru sink or captured terminal output.
- `logs/vllm/` and `logs/sglang/`: backend container-log dumps can contain sensitive diagnostics. They are not necessarily copies of launch-command logs.

MS-agent transcripts remain under scratch `output/.memory/`, and the episode record can retain container-log tails. The container's own `ms_agent.log` is written under its `/tmp` working directory, not automatically into host scratch. llama stderr uses a persistent `llama_stderr_*.log` file in Python's temporary directory. The launcher prints its exact path.

Do not publish raw logs, event streams, scratch directories, model provenance containing local details, or run databases without content review. Ignore rules do not make these files safe to share.

## Stop and resume

Press Ctrl+C in the foreground terminal to interrupt the command. This is not an atomic generation-boundary pause. Inspect outputs before resuming. Do not delete the archive, publication files, or training metadata if you intend to continue.

Use the same configuration, output paths, and printed run identifier:

```bash
uv run python scripts/run_train_loop.py \
	--config config/config-materials-adsorption-train.toml \
	--run-id '<printed-run-identifier>'
```

Replace `<printed-run-identifier>` with the actual identifier before running the command. Omitting `--run-id` starts a new run. It still uses the configured task archive. For an independent experiment, change the archive, blob, code-cache, generation, training-data, and adapter-output paths together.

Serving lineage identifies configured serving paths and ancestry. It does not attest the bytes loaded into GPU memory.

Generation-boundary recovery is distinct from optimizer-state recovery. No mid-fit GGUF checkpoint or optimizer-state resume is promised. There is no `--resume`, `--stop-after-generation`, or `--skip-training` switch. Do not infer safe boundary stopping from Ctrl+C timing.
