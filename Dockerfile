# Amazon Chronos-2 / Chronos-Bolt time-series forecasting API for Railway. Source: https://github.com/bon5co/chronos-railway
# CPU only (PyTorch), weights baked in, Bearer API_KEY, stateless (no volume needed).
# upstream: amazon-science/chronos-forecasting
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    FORECAST_MODEL_ROOT=/opt/models

COPY requirements.txt /tmp/requirements.txt
# chronos-forecasting==2.3.2 (newest release 2026-09-08) pulled with torch from the CPU wheel index.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(chronos|torch|transformers|numpy|pandas)"

# Bake Chronos-2 and the two Chronos-Bolt sizes (all Apache-2.0 per their Hugging Face model
# cards). Downloading at boot would put a fetch inside Railway's healthcheck window;
# downloading inside the first request would hit the edge's ~5 minute limit.
RUN python - <<'PY'
from huggingface_hub import snapshot_download
for name in ("chronos-2", "chronos-bolt-small", "chronos-bolt-base"):
    snapshot_download(f"amazon/{name}", local_dir=f"/opt/models/{name}",
                      allow_patterns=["*.json", "*.safetensors", "LICENSE*", "README.md"])
PY
RUN du -sh /opt/models/* && rm -rf /opt/hf /root/.cache

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin forecast \
    && chown -R 1000:0 /opt/models && chmod -R g+rwX /opt/models
WORKDIR /app
COPY app.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
USER 1000
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
