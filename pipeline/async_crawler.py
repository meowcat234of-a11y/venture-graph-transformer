import asyncio
import httpx

class AsyncCrawler:
    def __init__(self, rate_limit=5):
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def fetch(self, url, client):
        async with self.semaphore:
            try:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                return {"url": url, "content": response.text, "status": response.status_code}
            except Exception as e:
                return {"url": url, "error": str(e)}

    async def crawl(self, urls):
        async with httpx.AsyncClient() as client:
            tasks = [self.fetch(url, client) for url in urls]
            return await asyncio.gather(*tasks)

if __name__ == "__main__":
    urls = ["https://example.com", "https://example.org"]
    crawler = AsyncCrawler()
    results = asyncio.run(crawler.crawl(urls))
    print(results)
