import networkx as nx

def analyze_talent_flow(graph):
    """
    Analyzes talent flow between companies based on employment history.
    """
    flow_scores = {}
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'company':
            in_degree = graph.in_degree(node, weight='weight')
            out_degree = graph.out_degree(node, weight='weight')
            flow_scores[node] = {
                'inflow': in_degree,
                'outflow': out_degree,
                'net_flow': in_degree - out_degree
            }
    return flow_scores
