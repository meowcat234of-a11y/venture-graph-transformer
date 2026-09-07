import unittest

from analytics.momentum_scorer import calculate_momentum
from analytics.talent_flow import analyze_talent_flow
from pipeline.graph_builder import VentureGraph


class TestVentureAnalytics(unittest.TestCase):
    def test_weighted_talent_flow_and_momentum(self):
        venture_graph = VentureGraph()
        venture_graph.add_entity("Atlas", "company")
        venture_graph.add_entity("Beacon", "company")
        venture_graph.add_relationship("Atlas", "Beacon", "talent_move", weight=3.0)

        flow = analyze_talent_flow(venture_graph.graph)
        self.assertEqual(flow["Atlas"]["net_flow"], -3.0)
        self.assertEqual(flow["Beacon"]["net_flow"], 3.0)
        self.assertAlmostEqual(calculate_momentum(0.25, 3.0, 0.5), 1.4)


if __name__ == "__main__":
    unittest.main()
