import networkx as nx


class VentureGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_node(entity_id, type=entity_type, **attributes)

    def add_relationship(self, source, target, rel_type, weight=1.0):
        self.graph.add_edge(source, target, type=rel_type, weight=weight)

    def get_centrality(self):
        return nx.pagerank(self.graph)
