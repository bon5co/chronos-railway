#!/usr/bin/env python3
"""Amazon Chronos-2 / Chronos-Bolt zero-shot time-series forecasting HTTP API for Railway.

Upstream amazon-science/chronos-forecasting is a Python library: no server, no auth, and
weights that download from Hugging Face on first use. This file adds what a hosted deploy
needs:

  1. Bearer API_KEY on every route except GET /health. Forecasting burns CPU per request,
     so an unguarded public URL is somebody else's free compute.
  2. GET /health answers 503 until the default model is loaded and warm, 200 after, one
     path segment deep and unauthenticated, so Railway's healthcheck only passes a deploy
     that can actually answer a forecast.
  3. Weights are baked into the image (see Dockerfile); nothing downloads at boot or inside
     a request, where Railway's edge would abandon it at about 5 minutes.
  4. One JSON route, POST /v1/forecast: raw numbers in, quantile forecasts out, no
     DataFrame and no Python client needed. Chronos-2 also takes past and future covariates.
  5. Requests run one at a time behind a lock and torch is sized from the container's
     cgroup quota, not the host's.
"""

import math
import os
import socket
import sys
import threading
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY") or ""
MODEL_ROOT = os.environ.get("FORECAST_MODEL_ROOT") or "/opt/models"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL") or "chronos-2"
CPU_THREADS = int(os.environ.get("FORECAST_CPU_THREADS") or os.environ.get("OMP_NUM_THREADS") or 4)
MAX_SERIES = int(os.environ.get("MAX_SERIES") or 256)
MAX_POINTS = int(os.environ.get("MAX_POINTS") or 20_000)  # per series; the model reads its own context window
MAX_PREDICTION_LENGTH = int(os.environ.get("MAX_PREDICTION_LENGTH") or 1024)
MAX_BODY_MB = int(os.environ.get("MAX_BODY_MB") or 20)
PUBLIC_PATHS = {"/health"}

# id -> (Hugging Face repo, family, parameters in millions)
MODELS = {
    "chronos-2": ("amazon/chronos-2", "chronos2", 120),
    "chronos-bolt-small": ("amazon/chronos-bolt-small", "bolt", 48),
    "chronos-bolt-base": ("amazon/chronos-bolt-base", "bolt", 205),
}

app = FastAPI(
    title="Chronos time-series forecasting API",
    version="1.0.0",
    description="Zero-shot probabilistic forecasts from Amazon Chronos-2 and Chronos-Bolt. "
                "Authorization: Bearer <API_KEY>.",
)

_state = {"error": None, "loaded_in": None, "ready": False}
_pipes: dict = {}
_lock = threading.Lock()


def _pipeline(name: str):
    """Load a baked model from disk. The default stays resident; one other model is kept."""
    if name in _pipes:
        return _pipes[name]
    import torch
    from chronos import BaseChronosPipeline

    repo, _family, _params = MODELS[name]
    path = os.path.join(MODEL_ROOT, name)
    pipe = BaseChronosPipeline.from_pretrained(path, device_map="cpu", dtype=torch.float32)
    if name != DEFAULT_MODEL:
        for other in [k for k in _pipes if k != DEFAULT_MODEL]:
            del _pipes[other]
    _pipes[name] = pipe
    return pipe


def _load() -> None:
    t0 = time.time()
    try:
        import torch

        torch.set_num_threads(CPU_THREADS)
        pipe = _pipeline(DEFAULT_MODEL)
        # Warm the kernels so the first caller does not pay for them.
        rng = np.random.default_rng(0)
        warm = [np.sin(np.arange(128) / 6.0) + rng.normal(0, 0.05, 128)]
        _run(DEFAULT_MODEL, pipe, [np.asarray(w, dtype=np.float32) for w in warm], 12, [0.1, 0.5, 0.9], None, None, False)
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[chronos] ready model={DEFAULT_MODEL} threads={CPU_THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[chronos] LOAD FAILED: {_state['error']}", flush=True)


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        return JSONResponse(
            {"error": {"message": "Missing or invalid Authorization: Bearer <API_KEY>.", "type": "invalid_request_error"}},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_MB * 1024 * 1024:
        return JSONResponse({"error": {"message": f"body is larger than {MAX_BODY_MB} MB"}}, status_code=413)
    return await call_next(request)


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        body = {"status": "loading" if not _state["error"] else "failed", "error": _state["error"]}
        return JSONResponse(body, status_code=503)
    return {"status": "ok", "model": DEFAULT_MODEL, "threads": CPU_THREADS, "loaded_in_s": _state["loaded_in"]}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "amazon", "parameters_millions": v[2],
             "default": k == DEFAULT_MODEL, "covariates": v[1] == "chronos2"}
            for k, v in MODELS.items()
        ],
    }


class Series(BaseModel):
    id: Optional[str] = Field(None, description="label echoed back; defaults to series_<index>")
    values: list[Optional[float]] = Field(..., description="history, oldest first; null marks a missing value")
    past_covariates: Optional[dict[str, list[Optional[float]]]] = Field(
        None, description="chronos-2 only: name -> values, same length as `values`")
    future_covariates: Optional[dict[str, list[Optional[float]]]] = Field(
        None, description="chronos-2 only: name -> values, length prediction_length; names must also be in past_covariates")


class ForecastRequest(BaseModel):
    series: list[Series]
    prediction_length: int = Field(24, description="steps to forecast")
    quantiles: list[float] = Field([0.1, 0.5, 0.9], description="levels between 0 and 1")
    model: str = Field(DEFAULT_MODEL, description="chronos-2, chronos-bolt-small or chronos-bolt-base")
    cross_learning: bool = Field(False, description="chronos-2 only: share information across the series in one request")
    start: Optional[str] = Field(None, description="timestamp of the LAST history point, ISO 8601; with `freq` adds timestamps")
    freq: Optional[str] = Field(None, description="pandas frequency alias, e.g. D, h, 15min, MS; used with `start`")


def _bad(msg: str, code: int = 422):
    raise HTTPException(code, msg)


def _arr(values, what: str) -> np.ndarray:
    try:
        out = np.array([math.nan if v is None else v for v in values], dtype=np.float32)
    except (TypeError, ValueError):
        _bad(f"{what}: values must be numbers or null")
    if not np.isfinite(out[~np.isnan(out)]).all():
        _bad(f"{what}: infinite values are not allowed")
    return out


def _run(name, pipe, inputs, prediction_length, quantiles, past, future, cross_learning):
    """Return (quantile list [n x (H x Q)], median list [n x H]) as numpy."""
    family = MODELS[name][1]
    if family == "chronos2":
        if past is not None:
            batch = []
            for i, target in enumerate(inputs):
                item = {"target": target, "past_covariates": past[i]}
                if future is not None and future[i]:
                    item["future_covariates"] = future[i]
                batch.append(item)
        else:
            batch = inputs
        q, m = pipe.predict_quantiles(batch, prediction_length=prediction_length, quantile_levels=quantiles,
                                      cross_learning=cross_learning)
        # Chronos-2 returns (n_variates, H, Q) per series; one variate here.
        return [x[0].numpy() for x in q], [x[0].numpy() for x in m]
    import torch

    q, m = pipe.predict_quantiles([torch.from_numpy(x) for x in inputs], prediction_length=prediction_length,
                                  quantile_levels=quantiles)
    return [q[i].numpy() for i in range(q.shape[0])], [m[i].numpy() for i in range(m.shape[0])]


@app.post("/v1/forecast")
def forecast(req: ForecastRequest):
    if not _state["ready"]:
        raise HTTPException(503, _state["error"] or "model is still loading, retry in a few seconds")
    if req.model not in MODELS:
        _bad(f"model must be one of {', '.join(MODELS)}", 400)
    family = MODELS[req.model][1]
    if not 1 <= len(req.series) <= MAX_SERIES:
        _bad(f"send between 1 and {MAX_SERIES} series")
    if not 1 <= req.prediction_length <= MAX_PREDICTION_LENGTH:
        _bad(f"prediction_length must be between 1 and {MAX_PREDICTION_LENGTH}")
    if not req.quantiles or len(req.quantiles) > 20 or any(not 0.0 < q < 1.0 for q in req.quantiles):
        _bad("quantiles must be 1 to 20 levels strictly between 0 and 1")
    levels = sorted(set(round(float(q), 4) for q in req.quantiles))

    inputs, past, future = [], [], []
    uses_cov = any(s.past_covariates or s.future_covariates for s in req.series)
    if uses_cov and family != "chronos2":
        _bad("covariates are supported by chronos-2 only", 400)
    if uses_cov:
        keys_past = sorted((req.series[0].past_covariates or {}).keys())
        keys_future = sorted((req.series[0].future_covariates or {}).keys())
        if not set(keys_future) <= set(keys_past):
            _bad("every future_covariates name must also be in past_covariates")
    for i, s in enumerate(req.series):
        label = s.id or f"series_{i}"
        if not 3 <= len(s.values) <= MAX_POINTS:
            _bad(f"{label}: send between 3 and {MAX_POINTS} values")
        target = _arr(s.values, label)
        if np.isnan(target).all():
            _bad(f"{label}: every value is null")
        inputs.append(target)
        if uses_cov:
            pc, fc = s.past_covariates or {}, s.future_covariates or {}
            if sorted(pc.keys()) != keys_past or sorted(fc.keys()) != keys_future:
                _bad(f"{label}: every series must carry the same covariate names")
            for k, v in pc.items():
                if len(v) != len(target):
                    _bad(f"{label}: past_covariates.{k} must have {len(target)} values")
            for k, v in fc.items():
                if len(v) != req.prediction_length:
                    _bad(f"{label}: future_covariates.{k} must have {req.prediction_length} values")
            past.append({k: _arr(v, f"{label}.{k}") for k, v in pc.items()})
            future.append({k: _arr(v, f"{label}.{k}") for k, v in fc.items()})

    stamps = None
    if req.start or req.freq:
        if not (req.start and req.freq):
            _bad("send start and freq together")
        try:
            import pandas as pd

            rng = pd.date_range(start=req.start, periods=req.prediction_length + 1, freq=req.freq)
            stamps = [t.isoformat() for t in rng[1:]]
        except (ValueError, TypeError) as exc:
            _bad(f"start/freq not understood: {exc}")

    t0 = time.time()
    with _lock:
        try:
            pipe = _pipeline(req.model)
            qs, ms = _run(req.model, pipe, inputs, req.prediction_length, levels,
                          past if uses_cov else None, future if uses_cov else None, req.cross_learning)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"forecast failed: {type(exc).__name__}: {exc}") from exc
    took = time.time() - t0
    out = []
    for i, s in enumerate(req.series):
        item = {
            "id": s.id or f"series_{i}",
            "median": [round(float(x), 6) for x in ms[i]],
            "quantiles": {f"{lv:g}": [round(float(x), 6) for x in qs[i][:, j]] for j, lv in enumerate(levels)},
        }
        if stamps:
            item["timestamps"] = stamps
        out.append(item)
    return JSONResponse(
        {"model": req.model, "prediction_length": req.prediction_length, "quantile_levels": levels,
         "seconds": round(took, 3), "forecasts": out},
        headers={"X-Processing-Seconds": f"{took:.3f}"},
    )


def main() -> int:
    if not API_KEY:
        print("[chronos] FATAL: API_KEY is empty; refusing to publish /v1/forecast to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    if DEFAULT_MODEL not in MODELS:
        print(f"[chronos] FATAL: DEFAULT_MODEL must be one of {', '.join(MODELS)}", file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_load, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[chronos] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    config = uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "info"))
    uvicorn.Server(config).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
