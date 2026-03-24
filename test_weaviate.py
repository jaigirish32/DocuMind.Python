import asyncio
from DocuMind.core.logging.logger import setup_logging
from DocuMind.search.factory import create_weaviate_store

setup_logging(log_level="INFO", is_development=True)


async def test():
    store = create_weaviate_store()

    async with store._client:

        # Delete old collection first
        exists = await store._client.collections.exists("DocuMindChunk")
        if exists:
            await store._client.collections.delete("DocuMindChunk")
            print("Old collection deleted")

        # Recreate with new schema
        await store._ensure_collection()
        print("Collection ready: OK")

        # Check connection
        is_ready = await store._client.is_ready()
        print(f"Weaviate connected: {is_ready}")

        # List documents
        docs = await store.list_documents()
        print(f"Documents in index: {docs}")

    print()
    print("Weaviate connection OK")


asyncio.run(test())