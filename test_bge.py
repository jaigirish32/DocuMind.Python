import asyncio
#from DocuMind.bge.embedding_client import EmbeddingClient
from DocuMind.azure.embedding_client import EmbeddingClient
async def test():
    async with EmbeddingClient() as client:
        emb = await client.create_embeddings(["hello world"])
        print(len(emb[0]))

asyncio.run(test())