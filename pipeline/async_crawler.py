import asyncio

import httpx


class AsyncCrawler:
    def __init__(self, rate_limit=5, timeout=10.0, transport=None):
        if rate_limit < 1:
            raise ValueError("rate_limit must be positive")
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.timeout = timeout
        self.transport = transport

    async def fetch(self, url, client):
        async with self.semaphore:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return {"url": url, "content": response.text, "status": response.status_code}
            except httpx.HTTPError as exc:
                return {"url": url, "error": str(exc)}

    async def crawl(self, urls):
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            tasks = [self.fetch(url, client) for url in urls]
            return await asyncio.gather(*tasks)


if __name__ == "__main__":
    urls = ["https://example.com", "https://example.org"]
    crawler = AsyncCrawler()
    results = asyncio.run(crawler.crawl(urls))
    print(results)
