"""Uniform execution and failure mapping for all simulator backends."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import RLock
from time import perf_counter
from typing import Any, Iterable

from ..graph import CircuitGraph, ModelRef
from .backends import UnsupportedModelError
from .contracts import (
    Diagnostic,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
)


@dataclass(frozen=True)
class SimulationCacheKey:
    graph_hash: str
    backend_id: str
    backend_version: str
    semantic_request_hash: str


class SimulationResultCache:
    """Thread-safe exact cache for semantic simulation results."""

    def __init__(self, max_entries: int = 2048) -> None:
        if max_entries < 1:
            raise ValueError("simulation cache max_entries must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[SimulationCacheKey, SimulationResult] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: SimulationCacheKey) -> SimulationResult | None:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                self._misses += 1
                return None
            self._hits += 1
            self._values.move_to_end(key)
            return value

    def put(self, key: SimulationCacheKey, result: SimulationResult) -> None:
        if result.status in {SimulationStatus.TIMEOUT, SimulationStatus.UNAVAILABLE}:
            return
        with self._lock:
            self._values[key] = result
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._hits = 0
            self._misses = 0

    @property
    def statistics(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._values),
                "hits": self._hits,
                "misses": self._misses,
            }


_DEFAULT_SIMULATION_CACHE = SimulationResultCache()


class SimulationExecutor:
    def __init__(
        self,
        *,
        cache: SimulationResultCache | None = None,
        max_workers: int = 1,
        max_evaluations: int | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("simulation executor max_workers must be positive")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("simulation executor max_evaluations must be positive")
        self.cache = _DEFAULT_SIMULATION_CACHE if cache is None else cache
        self.max_workers = max_workers
        self.max_evaluations = max_evaluations

    def execute(
        self,
        backend: Any,
        graph: CircuitGraph,
        requests: Iterable[SimulationRequest],
        *,
        max_workers: int | None = None,
        max_evaluations: int | None = None,
    ) -> tuple[SimulationResult, ...]:
        requests = tuple(requests)
        if not requests:
            return ()
        workers = self.max_workers if max_workers is None else max_workers
        budget = self.max_evaluations if max_evaluations is None else max_evaluations
        if workers < 1:
            raise ValueError("simulation executor max_workers must be positive")
        if budget is not None and budget < 1:
            raise ValueError("simulation executor max_evaluations must be positive")
        available, reason = _availability(backend)
        if not available:
            return tuple(
                _failure_result(
                    backend,
                    graph,
                    request,
                    SimulationStatus.UNAVAILABLE,
                    "backend_unavailable",
                    reason,
                )
                for request in requests
            )
        try:
            compiled = backend.compile(graph)
        except UnsupportedModelError as exc:
            return tuple(
                _failure_result(
                    backend,
                    graph,
                    request,
                    SimulationStatus.UNSUPPORTED,
                    "unsupported_model",
                    str(exc),
                )
                for request in requests
            )
        except Exception as exc:
            return tuple(
                _failure_result(
                    backend,
                    graph,
                    request,
                    SimulationStatus.FAILED,
                    "compile_failed",
                    f"{type(exc).__name__}: {exc}",
                )
                for request in requests
            )

        backend_id, backend_version = _backend_identity(backend)
        results: list[SimulationResult | None] = [None] * len(requests)
        leaders: list[tuple[int, SimulationRequest, SimulationCacheKey]] = []
        followers: dict[int, list[int]] = {}
        leader_by_key: dict[SimulationCacheKey, int] = {}

        for index, request in enumerate(requests):
            key = SimulationCacheKey(
                graph_hash=graph.graph_hash,
                backend_id=backend_id,
                backend_version=backend_version,
                semantic_request_hash=request.semantic_hash,
            )
            if self.cache is not None:
                cached = self.cache.get(key)
                if cached is not None:
                    results[index] = _rebind_cached_result(cached, request)
                    continue
            leader_index = leader_by_key.get(key)
            if leader_index is not None:
                followers.setdefault(leader_index, []).append(index)
                continue
            leader_by_key[key] = index
            leaders.append((index, request, key))

        if budget is not None and len(leaders) > budget:
            allowed = leaders[:budget]
            blocked = leaders[budget:]
            for index, request, _ in blocked:
                blocked_result = _failure_result(
                    backend,
                    graph,
                    request,
                    SimulationStatus.FAILED,
                    "evaluation_budget_exhausted",
                    f"simulation evaluation budget of {budget} was exhausted",
                )
                results[index] = blocked_result
                for follower_index in followers.get(index, ()):
                    results[follower_index] = _rebind_cached_result(
                        blocked_result,
                        requests[follower_index],
                    )
            leaders = allowed

        def run(item: tuple[int, SimulationRequest, SimulationCacheKey]):
            index, request, key = item
            result = _execute_one(backend, compiled, graph, request)
            if self.cache is not None:
                self.cache.put(key, result)
            return index, request, result

        if workers > 1 and len(leaders) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                completed = list(pool.map(run, leaders))
        else:
            completed = [run(item) for item in leaders]

        for index, request, result in completed:
            results[index] = result
            for follower_index in followers.get(index, ()):
                results[follower_index] = _rebind_cached_result(result, requests[follower_index])

        assert all(result is not None for result in results)
        return tuple(result for result in results if result is not None)

    def execute_many(
        self,
        backend: Any,
        graph: CircuitGraph,
        requests: Iterable[SimulationRequest],
        **kwargs: Any,
    ) -> tuple[SimulationResult, ...]:
        """Explicit batch alias used by higher-level search and robustness code."""

        return self.execute(backend, graph, requests, **kwargs)

    def simulate_many(
        self,
        backend: Any,
        graph: CircuitGraph,
        requests: Iterable[SimulationRequest],
        **kwargs: Any,
    ) -> tuple[SimulationResult, ...]:
        return self.execute_many(backend, graph, requests, **kwargs)


def _execute_one(
    backend: Any,
    compiled: Any,
    graph: CircuitGraph,
    request: SimulationRequest,
) -> SimulationResult:
    started = perf_counter()
    try:
        result = backend.simulate(compiled, request)
    except Exception as exc:
        result = _failure_result(
            backend,
            graph,
            request,
            SimulationStatus.FAILED,
            "backend_exception",
            f"{type(exc).__name__}: {exc}",
            runtime_s=perf_counter() - started,
        )
    if result.request_id != request.request_id or result.request_hash != request.request_hash:
        result = _failure_result(
            backend,
            graph,
            request,
            SimulationStatus.FAILED,
            "result_contract_mismatch",
            "backend result does not reference the executed request",
            runtime_s=perf_counter() - started,
        )
    elapsed = perf_counter() - started
    if request.timeout_s is not None and elapsed > request.timeout_s and result.succeeded:
        result = _failure_result(
            backend,
            graph,
            request,
            SimulationStatus.TIMEOUT,
            "timeout",
            (
                f"backend completed in {elapsed:.6g} s after the "
                f"{request.timeout_s:.6g} s budget"
            ),
            runtime_s=elapsed,
        )
    return result


def _backend_identity(backend: Any) -> tuple[str, str]:
    capabilities = getattr(backend, "capabilities", None)
    capability_id = str(getattr(capabilities, "backend_id", type(backend).__name__))
    implementation_id = f"{type(backend).__module__}.{type(backend).__qualname__}"
    executable = str(getattr(backend, "executable", ""))
    backend_id = f"{capability_id}:{implementation_id}:{executable}"
    backend_version = str(
        getattr(backend, "backend_version", getattr(capabilities, "backend_version", "unknown"))
    )
    return backend_id, backend_version


def _rebind_cached_result(
    cached: SimulationResult,
    request: SimulationRequest,
) -> SimulationResult:
    statistics = dict(cached.statistics)
    statistics["cache_hit"] = True
    statistics["cache_source_evidence_hash"] = cached.evidence_hash
    return replace(
        cached,
        request_id=request.request_id,
        request_hash=request.request_hash,
        runtime_s=0.0,
        statistics=statistics,
    )


def _availability(backend: Any) -> tuple[bool, str]:
    probe = getattr(backend, "available", None)
    if not callable(probe):
        return True, "backend has no external availability requirement"
    try:
        value = probe()
    except Exception as exc:
        return False, f"availability probe failed: {type(exc).__name__}: {exc}"
    if isinstance(value, tuple):
        return bool(value[0]), str(value[1])
    return bool(value), "backend is available" if value else "backend is unavailable"


def _failure_result(
    backend: Any,
    graph: CircuitGraph,
    request: SimulationRequest,
    status: SimulationStatus,
    code: str,
    message: str,
    *,
    runtime_s: float = 0.0,
) -> SimulationResult:
    capabilities = getattr(backend, "capabilities", None)
    backend_id = str(getattr(capabilities, "backend_id", type(backend).__name__))
    backend_version = str(
        getattr(backend, "backend_version", getattr(capabilities, "backend_version", "unknown"))
    )
    return SimulationResult(
        request_id=request.request_id,
        request_hash=request.request_hash,
        graph_hash=graph.graph_hash,
        backend_id=backend_id,
        backend_version=backend_version,
        model_manifest=_model_manifest(graph),
        status=status,
        diagnostics=(Diagnostic(code, "error", message),),
        runtime_s=max(0.0, runtime_s),
        fidelity=request.fidelity,
    )


def _model_manifest(graph: CircuitGraph) -> tuple[ModelRef, ...]:
    unique = {
        (component.model.model_id, component.model.version): component.model
        for component in graph.components
    }
    return tuple(unique[key] for key in sorted(unique))
