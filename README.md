# Venture Graph Transformer

[![CI](https://github.com/meowcat234of-a11y/venture-graph-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/meowcat234of-a11y/venture-graph-transformer/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)

A systems-oriented prototype combining a decoder-only transformer with an
asynchronous relationship-graph pipeline and an interactive analytics dashboard.

## What it demonstrates

- A decoder built from RMSNorm, rotary position embeddings, SwiGLU blocks, and
  causal multi-head attention.
- Stateful key-value caching with tests against full-sequence decoding.
- Bounded-concurrency HTTP collection and deterministic entity extraction.
- Directed relationship graphs, weighted talent-flow metrics, and PageRank.
- A Streamlit dashboard for ranking company momentum from uploaded edge lists.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover tests
```

Launch the dashboard with:

```bash
streamlit run dashboard/app.py
```

Upload a CSV containing `source`, `target`, and an optional numeric `weight`
column. The app uses a small built-in graph when no file is supplied.

## Decoder notes

RoPE applies a position-dependent rotation to each query and key, while the
feed-forward block uses

$$
\operatorname{SwiGLU}(x)=
W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).
$$

During incremental generation, every layer retains prior keys and values so
only the newest token needs to be projected.

## Repository layout

```text
model/       decoder architecture, tokenizer, and KV cache
pipeline/    asynchronous collection and graph construction
analytics/   momentum and talent-flow metrics
dashboard/   Streamlit application and input validation
tests/       deterministic unit tests
```

## Scope

The decoder is intentionally untrained; generated tokens validate architecture
and inference mechanics rather than language quality. The crawler and entity
extractor are research components, not a production ingestion service.
