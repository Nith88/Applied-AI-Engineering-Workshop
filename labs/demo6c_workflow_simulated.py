from workflow_core import build_workflow, run_one

graph = build_workflow()

sample_emails = [
    {
        "sender": "ops@example.com",
        "subject": "URGENT: production server down",
        "body": "Production is unavailable. Please investigate immediately.",
    },
    {
        "sender": "billing@example.com",
        "subject": "Invoice confirmation required",
        "body": "Could you please confirm that you received the invoice?",
    },
    {
        "sender": "receipts@example.com",
        "subject": "Payment receipt",
        "body": "This is an automated receipt for your recent payment. No action is required.",
    },
]

for email_data in sample_emails:
    run_one(graph, email_data)

print("\nGRAPH DESCRIPTION (Mermaid):")
print(graph.get_graph().draw_mermaid())

print("\nSTREAM MODE — one example:")
run_one(graph, sample_emails[1], stream=True)
