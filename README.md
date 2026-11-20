# chronos-railway

Zero-shot time-series forecasting API around [Amazon Chronos-2](https://huggingface.co/amazon/chronos-2) and [Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small) ([amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) 2.3.2), run on CPU with PyTorch. For Railway. Weights baked into the image, Bearer `API_KEY` on every route except `GET /health`.

- `POST /v1/forecast` JSON `{"series":[{"id","values":[...]}], "prediction_length":24, "quantiles":[0.1,0.5,0.9], "model":"chronos-2"}` -> `forecasts[]` with `median` and `quantiles`. Optional `start` + `freq` add timestamps; `chronos-2` also takes `past_covariates` / `future_covariates` and `cross_learning`.
- `GET /v1/models`, `GET /health` (503 until the model is loaded, then 200).
- Env: `API_KEY` (required), `DEFAULT_MODEL` (`chronos-2`), `MAX_SERIES` (256), `MAX_POINTS` (20000), `MAX_PREDICTION_LENGTH` (1024), `FORECAST_CPU_THREADS` (from the cgroup quota).

Wrapper code is Apache-2.0. chronos-forecasting is Apache-2.0. Weights `amazon/chronos-2`, `amazon/chronos-bolt-small` and `amazon/chronos-bolt-base` are Apache-2.0 (license field of each Hugging Face model card).
