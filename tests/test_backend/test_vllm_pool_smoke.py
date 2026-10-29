from src.backend.vllm import _build_pool_smoke_test


class _FakeSession:
    def __init__(self, name: str):
        self.smoke_test = lambda: f"ok-{name}"


class _FakePool:
    def __init__(self, endpoint_ids: list[str]):
        self._endpoint_ids = endpoint_ids
        self.unhealthy: list[str] = []

    def endpoint_ids(self) -> list[str]:
        return list(self._endpoint_ids)

    def is_healthy(self, endpoint_id: str) -> bool:
        return endpoint_id not in self.unhealthy

    def mark_unhealthy(self, endpoint_id: str, _reason: str) -> None:
        self.unhealthy.append(endpoint_id)


def test_pool_smoke_test_snapshots_primary_callback() -> None:
    sessions = {
        "primary": _FakeSession("primary"),
        "secondary": _FakeSession("secondary"),
    }
    pool = _FakePool(["primary", "secondary"])
    run = _build_pool_smoke_test(pool, sessions)  # type: ignore[arg-type]

    sessions["primary"].smoke_test = run

    assert run() == "primary: ok-primary"


def test_pool_smoke_test_falls_through_after_primary_failure() -> None:
    def fail() -> str:
        raise RuntimeError("down")

    sessions = {
        "primary": _FakeSession("primary"),
        "secondary": _FakeSession("secondary"),
    }
    sessions["primary"].smoke_test = fail
    pool = _FakePool(["primary", "secondary"])
    run = _build_pool_smoke_test(pool, sessions)  # type: ignore[arg-type]

    sessions["primary"].smoke_test = run

    assert run() == "secondary: ok-secondary"
    assert pool.unhealthy == ["primary"]
