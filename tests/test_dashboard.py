import unittest

import pandas as pd

from analytics.talent_flow import analyze_talent_flow
from dashboard.graph import build_graph


class TestDashboardGraph(unittest.TestCase):
    def test_uploaded_edges_contribute_to_talent_flow(self):
        edges = pd.DataFrame({
            "source": ["Atlas"],
            "target": ["Beacon"],
            "weight": [3.0],
        })

        graph = build_graph(edges)
        flow = analyze_talent_flow(graph)

        self.assertEqual(flow["Atlas"]["net_flow"], -3.0)
        self.assertEqual(flow["Beacon"]["net_flow"], 3.0)


if __name__ == "__main__":
    unittest.main()
