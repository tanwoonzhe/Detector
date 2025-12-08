import asyncio
import aiohttp

URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
# 注意：免费版不要传 interval=hourly，2-90 天会自动返回小时粒度
PARAMS = {"vs_currency": "usd", "days": 90}

async def main():
    timeout = aiohttp.ClientTimeout(total=15)
    headers = {"User-Agent": "btc-predictor-test/1.0", "Accept": "application/json"}
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
