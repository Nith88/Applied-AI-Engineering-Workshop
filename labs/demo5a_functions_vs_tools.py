"""LU07 Demo: Compare a normal Python function with a LangChain tool."""

from langchain_core.tools import tool

emails = [
    {"id": 1, "sender": "finance@example.com",
     "subject": "Final Invoice", "body": "Please review the final invoice."},
    {"id": 2, "sender": "hr@example.com",
     "subject": "Workshop Schedule", "body": "The workshop starts Monday."},
    {"id": 3, "sender": "finance@example.com",
     "subject": "Invoice Payment Reminder", "body": "Please confirm payment status."},
    {"id": 4, "sender": "manager@example.com",
     "subject": "Project Meeting", "body": "Let us meet tomorrow."},
]

# A normal function: the developer explicitly calls it.
def search_emails_function(query: str) -> list[dict]:
    return [
        item for item in emails
        if query.casefold() in item["subject"].casefold()
    ]

print("--- NORMAL FUNCTION ---")
print(search_emails_function("invoice"))


# A tool: the model can inspect its name, description and input schema.
@tool
def search_emails_tool(query: str) -> str:
    """Search the demo inbox by subject keyword. Use when the user wants to find emails."""
    matches = search_emails_function(query)
    if not matches:
        return "No matching emails found."
    return "\n\n".join(
        f"ID: {m['id']}\nFrom: {m['sender']}\nSubject: {m['subject']}\nBody: {m['body']}"
        for m in matches
    )

print("\n--- TOOL INVOKED MANUALLY FOR TESTING ---")
print(search_emails_tool.invoke({"query": "invoice"}))

print("\n--- TOOL METADATA SHOWN TO A MODEL ---")
print("Name:", search_emails_tool.name)
print("Description:", search_emails_tool.description)
print("Input schema:", search_emails_tool.args)
