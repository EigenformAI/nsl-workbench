def infer_tool_call_parser(engine: str, model: str) -> str | None:
    normalized = model.strip().lower()
    if engine == "vllm":
        if "qwen" in normalized or "qwq" in normalized:
            return "hermes"
        return None

    if engine == "sglang":
        if "qwen" in normalized or "qwq" in normalized:
            return "qwen25"
        if "llama-3" in normalized or "llama3" in normalized:
            return "llama3"
        if "gemma-3" in normalized or "gemma3" in normalized:
            return "gemma3"
        if "deepseek-v3" in normalized:
            return "deepseekv3"
        if "mistral" in normalized:
            return "mistral"
        return None

    raise ValueError(f"Unknown tool parser engine: {engine}")
