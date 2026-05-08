"""
Wipe all documents from the Azure AI Search index.
Keeps the index schema intact — only deletes documents.

Usage:
    cd C:\\mywork\\DocuMind.Python
    .\\.venv312\\Scripts\\Activate.ps1
    python wipe_search_index.py
"""
import os
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
key        = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

if not all([endpoint, key, index_name]):
    print("[ERROR] Missing AZURE_SEARCH_* env vars. Check your .env file.")
    raise SystemExit(1)

print(f"[INFO] Endpoint:   {endpoint}")
print(f"[INFO] Index:      {index_name}")

credential    = AzureKeyCredential(key)
index_client  = SearchIndexClient(endpoint=endpoint, credential=credential)
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# Auto-discover the key field from the index schema
index = index_client.get_index(index_name)
key_field = next(f.name for f in index.fields if f.key)
print(f"[INFO] Key field:  {key_field}")

# Pull all document keys (paginate in case index has > 1000 docs)
all_keys = []
results = search_client.search(search_text="*", select=[key_field], top=1000)
for doc in results:
    all_keys.append(doc[key_field])

print(f"[INFO] Found {len(all_keys)} documents in index")

if not all_keys:
    print("[OK] Index is already empty.")
    raise SystemExit(0)

# Confirm before destruction
confirm = input(f"\nDelete ALL {len(all_keys)} documents? Type 'yes' to confirm: ")
if confirm.strip().lower() != "yes":
    print("[CANCELLED] No changes made.")
    raise SystemExit(0)

# Delete in batches of 1000 (Azure Search limit)
deleted_total = 0
for i in range(0, len(all_keys), 1000):
    batch = all_keys[i:i+1000]
    docs_to_delete = [{key_field: k} for k in batch]
    result = search_client.delete_documents(documents=docs_to_delete)
    succeeded = sum(1 for r in result if r.succeeded)
    failed    = len(result) - succeeded
    deleted_total += succeeded
    print(f"[OK] Batch {i//1000 + 1}: deleted {succeeded}, failed {failed}")

print(f"\n[DONE] Deleted {deleted_total}/{len(all_keys)} documents.")
print("[NOTE] Index schema is unchanged. Ready for fresh uploads.")
