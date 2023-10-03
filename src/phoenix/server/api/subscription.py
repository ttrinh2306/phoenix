import asyncio
from typing import AsyncGenerator

import strawberry


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def trace_count(self) -> AsyncGenerator[int, None]:
        for i in range(100):
            yield i
            await asyncio.sleep(0.5)
            yield i
            await asyncio.sleep(0.5)
