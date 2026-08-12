import asyncio

from aclimate_agent import AClimateAgent


async def main():
    agent = AClimateAgent()

    response = await agent.chat(
        "Está CONDAGUA en los sitios disponibles?"
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())