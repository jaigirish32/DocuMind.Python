import asyncio
from pathlib import Path

from DocuMind.core.logging.logger import setup_logging
from DocuMind.bge.embedding_client import EmbeddingClient   # ✅ CHANGED
from DocuMind.azure.chat_client import ChatClient
from DocuMind.search.factory import create_weaviate_store
from DocuMind.agents.documind_agent import DocuMindAgent
from DocuMind.documents.readers.pdf_reader import PdfReader
from DocuMind.documents.indexing.document_indexer import DocumentIndexer

setup_logging(log_level="INFO", is_development=True)

TESLA_PDF       = Path(r"C:\mywork\samplePDFS\tesla-10k-2023.pdf")
ELECTRICITY_PDF = Path(r"C:\mywork\samplePDFS\ElectricityBill.pdf")

# 🔥 NEW FLAG
FORCE_REINDEX = True


async def ensure_indexed(
    store:       object,
    embedder:    object,
    path:        Path,
    document_id: str,
    boilerplate: list[str] = None,
) -> None:
    """Index document with optional force re-index."""

    docs = await store.list_documents()

    
    if path.name in docs:
        if FORCE_REINDEX:
            print(f"Re-indexing (deleting old): {path.name}")
            await store.delete_document(document_id)
        else:
            print(f"Already indexed: {path.name} — skipping")
            return

    print(f"Indexing: {path.name}...")

    reader  = PdfReader()
    indexer = DocumentIndexer(
        reader   = reader,
        embedder = embedder,
        store    = store,
    )

    await indexer.index(
        path                 = path,
        document_id          = document_id,
        boilerplate_patterns = boilerplate or [],
    )

    print(f"Indexed: {path.name}")


async def test():

    async with create_weaviate_store() as store:
        async with EmbeddingClient() as embedder:

            print("Deleting existing collection (for BGE switch)...")
            try:
                await store._client.collections.delete("DocuMindChunk")
                print("Collection deleted")
            except Exception:
                print("Collection may not exist, continuing...")
            
            await ensure_indexed(
                store       = store,
                embedder    = embedder,
                path        = TESLA_PDF,
                document_id = "tesla-2023",
                boilerplate = ["sec.gov/archives"],
            )

            await ensure_indexed(
                store       = store,
                embedder    = embedder,
                path        = ELECTRICITY_PDF,
                document_id = "electricity-bill",
            )

            print()

            async with ChatClient() as chat:
                agent = DocuMindAgent(
                    embedder = embedder,
                    chat     = chat,
                    store    = store,
                )

                # Tesla questions
                print("═" * 50)
                print("Tesla 10-K questions")
                print("═" * 50)

                tesla_questions = [
                    "what was Tesla total revenue in 2023?",
                    "what are the main risk factors?",
                    "who is the CEO of Tesla?",
                ]

                for question in tesla_questions:
                    print(f"\n── {question}")
                    result = await agent.ask_structured(
                        question    = question,
                        document_id = "tesla-2023",
                    )
                    print(f"   Pages  : {result.source_pages}")
                    print(f"   Answer : {result.answer[:200]}")

                print()

                # Electricity bill questions
                print("═" * 50)
                print("Electricity bill questions")
                print("═" * 50)

                electricity_questions = [
                    "what is the grand total amount?",
                    "what is the flat number?",
                    "what is the meter number?",
                    "what is the total recharge amount for the month?",
                    "what are the fixed charges?",
                ]

                for question in electricity_questions:
                    print(f"\n── {question}")
                    result = await agent.ask_structured(
                        question    = question,
                        document_id = "electricity-bill",
                    )
                    print(f"   Pages  : {result.source_pages}")
                    print(f"   Answer : {result.answer[:200]}")


asyncio.run(test())