import networkx as nx


def build_graph(edges):
    required = {"source", "target"}
    if not required.issubset(edges.columns):
        raise ValueError("CSV must include source and target columns")
    if edges.empty:
        raise ValueError("CSV must contain at least one edge")

    graph = nx.DiGraph()
    for edge in edges.itertuples(index=False):
        weight = float(getattr(edge, "weight", 1.0))
        graph.add_node(edge.source, type="company")
        graph.add_node(edge.target, type="company")
        graph.add_edge(edge.source, edge.target, weight=weight)
    return graph
