import unittest

from pipeline.entity_extractor import EntityExtractor


class TestEntityExtractor(unittest.TestCase):
    def test_entities_are_unique_and_deterministic(self):
        entities = EntityExtractor().extract("Beacon Labs met Atlas Corp and Beacon Labs")

        self.assertEqual(entities["company"], ["Atlas Corp", "Beacon Labs"])
        self.assertEqual(entities["founder"], ["Atlas Corp", "Beacon Labs"])


if __name__ == "__main__":
    unittest.main()
