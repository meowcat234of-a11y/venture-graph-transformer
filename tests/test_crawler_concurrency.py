import unittest
import asyncio
from pipeline.async_crawler import AsyncCrawler

class TestAsyncCrawler(unittest.TestCase):
    def test_crawl(self):
        crawler = AsyncCrawler(rate_limit=2)
        urls = ["http://example.com"]
        
        results = asyncio.run(crawler.crawl(urls))
        self.assertEqual(len(results), 1)
        self.assertIn("url", results[0])

if __name__ == "__main__":
    unittest.main()
