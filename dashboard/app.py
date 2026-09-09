import networkx as nx
import pandas as pd
import streamlit as st

from analytics.momentum_scorer import calculate_momentum
from analytics.talent_flow import analyze_talent_flow
from dashboard.graph import build_graph

st.set_page_config(page_title="Venture Graph Transformer", layout="wide")
st.title("Venture Graph Transformer")
st.caption(
    "Rank company momentum from relationship centrality, talent flow, and analyst sentiment."
)

uploaded = st.file_uploader("Upload venture edges as CSV", type="csv")
if uploaded is None:
    edges = pd.DataFrame(
        {
            "source": ["Atlas", "Beacon", "Atlas", "Cinder"],
            "target": ["Beacon", "Cinder", "Cinder", "Atlas"],
            "weight": [2.0, 1.0, 3.0, 1.0],
        }
    )
    st.info(
        "Showing a small demo graph. Upload a CSV with source, target, and optional weight columns to analyze your own data."
    )
else:
    edges = pd.read_csv(uploaded)

try:
    graph = build_graph(edges)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

centrality = nx.pagerank(graph, weight="weight")
flows = analyze_talent_flow(graph)
sentiment = st.slider("Analyst sentiment", 0.0, 1.0, 0.5, 0.05)
scores = {
    company: calculate_momentum(
        centrality[company], flows.get(company, {}).get("net_flow", 0.0), sentiment
    )
    for company in graph.nodes
}
results = pd.DataFrame(
    {
        "company": list(graph.nodes),
        "pagerank": [centrality[company] for company in graph.nodes],
        "net_talent_flow": [flows.get(company, {}).get("net_flow", 0.0) for company in graph.nodes],
        "momentum": [scores[company] for company in graph.nodes],
    }
).sort_values("momentum", ascending=False)

top_company = results.iloc[0]
left, middle, right = st.columns(3)
left.metric("Companies", len(graph.nodes))
middle.metric("Relationships", len(graph.edges))
right.metric("Top momentum", f"{top_company['momentum']:.3f}", top_company["company"])
st.dataframe(results, hide_index=True, use_container_width=True)
st.bar_chart(results.set_index("company")["momentum"])
