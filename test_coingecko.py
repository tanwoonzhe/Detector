import asyncio
import aiohttp

URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
PARAMS = {"vs_currency": "usd", "days": 90, "interval": "hourly"}

async def main():
    timeout = aiohttp.ClientTimeout(total=15)
    headers = {"User-Agent": "btc-predictor-test/1.0"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        try:
            async with session.get(URL, params=PARAMS) as resp:
                print("status:", resp.status)
                text = await resp.text()
                print("response head:", text[:500])
        except Exception as e:
            print("error:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
