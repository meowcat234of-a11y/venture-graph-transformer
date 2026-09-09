# Venture Graph Transformer

[![CI](https://github.com/meowcat234of-a11y/venture-graph-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/meowcat234of-a11y/venture-graph-transformer/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)

This repo grew out of two things I wanted to understand by building them: the
inside of a decoder-only transformer and the mechanics of turning messy
relationship data into a useful graph. The result is part model exercise, part
data pipeline, with a small Streamlit app for exploring the output.

## What's included

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

## What I focused on in the decoder

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

## What to expect

The decoder is intentionally untrained. Its generated tokens are only a check
of the architecture and inference path, not a demonstration of language
quality. The crawler and entity extractor are also research components; they
will need more defensive handling before they are suitable for a production
ingestion service.
