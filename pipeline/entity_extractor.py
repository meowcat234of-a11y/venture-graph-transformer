import re


class EntityExtractor:
    def __init__(self):
        self.patterns = {
            "company": r"\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)*)\b",
            "founder": r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b",
        }

    def extract(self, text):
        entities = {}
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = sorted(set(matches))
        return entities
