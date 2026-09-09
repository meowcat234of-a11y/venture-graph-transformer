import asyncio
import unittest

import httpx

from pipeline.async_crawler import AsyncCrawler


class TestAsyncCrawler(unittest.TestCase):
    def test_crawl(self):
        seen_timeouts = []

        def respond(request):
            seen_timeouts.append(request.extensions["timeout"]["read"])
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(respond)
        crawler = AsyncCrawler(rate_limit=2, timeout=0.25, transport=transport)
        urls = ["http://example.com"]

        results = asyncio.run(crawler.crawl(urls))
        self.assertEqual(len(results), 1)
        self.assertIn("url", results[0])
        self.assertEqual(seen_timeouts, [0.25])


if __name__ == "__main__":
    unittest.main()
