"""
scripts/seed_demo_emails.py
Creates realistic demo emails and indexes them into Weaviate.
Run once to populate demo data for client presentations.

Usage:
    cd C:\mywork\DocuMind.Python
    .venv\Scripts\activate
    python scripts/seed_demo_emails.py
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DocuMind.email.models import EmailMessage
from DocuMind.email.indexer import EmailIndexer
from DocuMind.bge.embedding_client import EmbeddingClient
from DocuMind.search.factory import create_weaviate_store


DEMO_EMAILS = [
    EmailMessage(
        message_id  = "demo_001",
        thread_id   = "thread_contract_001",
        subject     = "Contract Agreement — Acme Corp Software License",
        sender      = "sarah.johnson@acmecorp.com",
        recipients  = ["you@company.com"],
        body        = """Hi,

Please find below the agreed terms for the software license contract:

- License fee: $45,000 per year
- Payment terms: Net 30 days
- Contract duration: 3 years with annual renewal option
- Support: 24/7 premium support included
- Users: Up to 500 named users

We are happy to proceed on these terms. Please confirm by end of week.

Best regards,
Sarah Johnson
Head of Procurement, Acme Corp""",
        date        = datetime(2024, 3, 1, 9, 30),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_002",
        thread_id   = "thread_contract_001",
        subject     = "Re: Contract Agreement — Acme Corp Software License",
        sender      = "legal@acmecorp.com",
        recipients  = ["you@company.com"],
        body        = """Following up on the contract terms.

Our legal team has reviewed the agreement and requests the following changes:

1. Payment terms changed from Net 30 to Net 45 days
2. Add a data protection clause (GDPR compliance)
3. Liability cap to be set at 2x annual license fee ($90,000)
4. Termination notice period: 90 days

Please confirm these amendments and we can proceed to signing.

regards,
Legal Team, Acme Corp""",
        date        = datetime(2024, 3, 5, 14, 0),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_003",
        thread_id   = "thread_invoice_001",
        subject     = "Invoice #1042 — Payment Due March 30",
        sender      = "accounts@vendorcorp.com",
        recipients  = ["finance@company.com"],
        body        = """Dear Finance Team,

Please find invoice details below:

Invoice Number: #1042
Invoice Date: March 1, 2024
Due Date: March 30, 2024

Services:
- Cloud infrastructure setup: $12,000
- Security audit: $5,000
- Training (2 days): $3,000

Total Amount Due: $20,000

Payment details:
Bank: First National Bank
Account: 1234567890
Routing: 021000021

Please confirm receipt of this invoice.

Best,
Accounts Team
VendorCorp""",
        date        = datetime(2024, 3, 1, 10, 0),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_004",
        thread_id   = "thread_invoice_001",
        subject     = "Re: Invoice #1042 — Payment Confirmation",
        sender      = "finance@company.com",
        recipients  = ["accounts@vendorcorp.com"],
        body        = """Hi,

We confirm receipt of Invoice #1042 for $20,000.

Payment has been scheduled for March 28, 2024 (2 days before due date).
Transaction reference: TXN-2024-03-28-1042

Please confirm once payment is received.

Thanks,
Finance Team""",
        date        = datetime(2024, 3, 15, 11, 30),
        labels      = ["SENT"],
    ),
    EmailMessage(
        message_id  = "demo_005",
        thread_id   = "thread_project_001",
        subject     = "Project Alpha — Status Update Week 10",
        sender      = "pm@company.com",
        recipients  = ["team@company.com", "stakeholders@company.com"],
        body        = """Team,

Here is the Week 10 status update for Project Alpha:

COMPLETED THIS WEEK:
- User authentication module — 100% complete
- Database schema migration — complete
- API endpoints for dashboard — complete

IN PROGRESS:
- Frontend React components — 65% complete (on track)
- Payment integration — 40% complete (slight delay)
- Testing and QA — starting next week

BLOCKERS:
- Payment gateway API keys pending from finance team (blocking payment integration)
- Need design approval for the mobile UI by Friday

NEXT WEEK PLAN:
- Complete frontend components
- Begin QA testing
- Resolve payment gateway blocker

Overall project status: ON TRACK
Estimated completion: April 15, 2024

Please raise any concerns before Friday's standup.

Regards,
Project Manager""",
        date        = datetime(2024, 3, 8, 9, 0),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_006",
        thread_id   = "thread_meeting_001",
        subject     = "Meeting Notes — Q1 Strategy Review",
        sender      = "ceo@company.com",
        recipients  = ["leadership@company.com"],
        body        = """All,

Please find the key decisions and action items from today's Q1 Strategy Review:

DECISIONS MADE:
1. Expand into European market by Q3 2024
2. Hire 15 new engineers by end of Q2
3. Increase marketing budget by 30% for product launch
4. Partner with CloudTech for infrastructure

ACTION ITEMS:
- John: Prepare European market entry plan by March 20
- Sarah: Begin recruitment drive, post 15 job listings by March 15
- Mike: Finalise CloudTech partnership agreement by April 1
- Finance: Revise Q2 budget to reflect 30% marketing increase

NEXT MEETING:
Q2 Strategy Review — June 1, 2024

Best,
CEO""",
        date        = datetime(2024, 3, 10, 16, 0),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_007",
        thread_id   = "thread_support_001",
        subject     = "Critical Bug — Production System Down",
        sender      = "alerts@monitoring.com",
        recipients  = ["engineering@company.com"],
        body        = """CRITICAL ALERT

Production system outage detected at 14:32 UTC.

Affected services:
- API Gateway: DOWN
- User authentication: DOWN
- Dashboard: Partially down

Error logs show database connection timeout.
Root cause: Database server ran out of connections (max 100, current 100).

Immediate action required:
1. Increase max connections to 500
2. Restart connection pool
3. Monitor for 30 minutes

Severity: P0 — Critical
Incident ID: INC-2024-0312
On-call engineer: DevOps Team""",
        date        = datetime(2024, 3, 12, 14, 35),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_008",
        thread_id   = "thread_hr_001",
        subject     = "Job Offer — Senior Software Engineer",
        sender      = "hr@company.com",
        recipients  = ["candidate@gmail.com"],
        body        = """Dear Alex,

We are pleased to offer you the position of Senior Software Engineer.

Offer Details:
- Position: Senior Software Engineer
- Start Date: April 1, 2024
- Salary: $145,000 per year
- Bonus: Up to 15% annual performance bonus
- Stock Options: 10,000 options vesting over 4 years
- Benefits: Full health, dental, vision coverage
- Remote: Hybrid (3 days office, 2 days remote)

Please sign and return the offer letter by March 20, 2024.

We look forward to having you on the team.

Best regards,
HR Team""",
        date        = datetime(2024, 3, 13, 10, 0),
        labels      = ["SENT"],
    ),
    EmailMessage(
        message_id  = "demo_009",
        thread_id   = "thread_vendor_001",
        subject     = "Vendor Proposal — Cloud Storage Solution",
        sender      = "sales@cloudstore.com",
        recipients  = ["it@company.com"],
        body        = """Hi,

Thank you for your interest in CloudStore solutions.

Our proposed solution for your 50TB storage requirement:

PLAN: Enterprise Pro
- Storage: 100TB (scalable to 1PB)
- Redundancy: 3x geographic replication
- Uptime SLA: 99.99%
- Security: AES-256 encryption, SOC2 certified
- Support: Dedicated account manager

PRICING:
- Monthly: $2,500/month
- Annual (20% discount): $24,000/year
- Setup fee: Waived for annual plan

COMPARISON vs current solution:
- 40% cheaper than AWS S3 for your usage
- 3x faster upload speeds
- Better compliance for GDPR

Free 30-day trial available. Shall we schedule a demo?

Best,
Sales Team, CloudStore""",
        date        = datetime(2024, 3, 14, 11, 0),
        labels      = ["INBOX"],
    ),
    EmailMessage(
        message_id  = "demo_010",
        thread_id   = "thread_legal_001",
        subject     = "NDA Required — Partnership Discussion with TechVentures",
        sender      = "legal@company.com",
        recipients  = ["ceo@company.com", "bd@company.com"],
        body        = """Hi,

Before we proceed with partnership discussions with TechVentures, 
we need to execute a mutual NDA.

NDA Key Terms:
- Duration: 2 years
- Scope: All confidential information shared during partnership discussions
- Exclusions: Publicly available information, independently developed info
- Governing law: Delaware, USA

TechVentures has agreed to our standard NDA template.
Signing scheduled for March 18, 2024.

Action needed:
- CEO signature required by March 17
- BD team to prepare discussion agenda after NDA is signed

Please let me know if you have any questions.

Legal Team""",
        date        = datetime(2024, 3, 15, 9, 0),
        labels      = ["INBOX"],
    ),
]


async def main():
    print("Connecting to Weaviate...")
    store = create_weaviate_store()
    await store.__aenter__()

    print("Initialising embedding client...")
    embedder = EmbeddingClient()

    indexer = EmailIndexer(embedder=embedder, store=store)

    print(f"Indexing {len(DEMO_EMAILS)} demo emails...")
    result = await indexer.index(DEMO_EMAILS)

    print(f"Done! Indexed {result['emails']} emails → {result['chunks']} chunks")
    await store._client.close()


if __name__ == "__main__":
    asyncio.run(main())