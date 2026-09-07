# Venture Graph Transformer

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#) [![NetworkX](https://img.shields.io/badge/NetworkX-graph%20analytics-2ea44f)](#)

A dual-layer system combining a from-scratch decoder-only transformer with an asynchronous graph pipeline to ingest, parse, and score startup momentum.

## Scope

The decoder validates architecture and inference mechanics, including cache-equivalent decoding. It is untrained; generated tokens are diagnostic outputs rather than meaningful language-model completions.

## Setup
```bash
pip install -e .
```

## Structure
- `model/`: From-scratch transformer architecture with RoPE, SwiGLU, RMSNorm
- `pipeline/`: Async graph crawling and entity extraction
- `analytics/`: Momentum scoring and talent flow analysis
- `dashboard/`: Streamlit dashboard
- `tests/`: Basic structural tests

## Decoder architecture

RoPE applies a position-dependent rotation to queries and keys:

$$\operatorname{RoPE}(x_m)=x_m e^{im\theta}.$$

The feed-forward block uses SwiGLU:

$$\operatorname{SwiGLU}(x)=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).$$

During generation, each layer retains its past keys and values so only the newest token is projected at each decoding step.

## Testing

Install with `pip install -e .` and run `python -m unittest discover tests`.

## Dashboard

Launch `streamlit run dashboard/app.py` to explore the included demo graph or upload an edge-list CSV with `source`, `target`, and optional `weight` columns.
