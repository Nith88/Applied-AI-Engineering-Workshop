from workflow_core import USE_GEMINI, llm, classify_from_demo, make_initial_state

email_data = {
    "sender": "billing@example.com",
    "subject": "Please confirm the invoice",
    "body": "Could you please confirm that you received the invoice?",
}
state = make_initial_state(email_data)

print("ONE-SHOT CLASSIFICATION")
if USE_GEMINI:
    from workflow_core import classify_with_gemini
    result = classify_with_gemini(state)
else:
    result = classify_from_demo(state)

print(result)
print("\nNotice: this gives us classification data, but it does not yet route to a workflow outcome.")
