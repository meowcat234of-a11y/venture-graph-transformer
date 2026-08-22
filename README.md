# Venture Graph Transformer

A dual-layer system combining a from-scratch decoder-only transformer with an asynchronous graph pipeline to ingest, parse, and score startup momentum.

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
