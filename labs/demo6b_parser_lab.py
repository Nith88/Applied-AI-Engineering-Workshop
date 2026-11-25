from workflow_core import validate_classification, _extract_json

# The lesson's simple line parser is intentionally reproduced here.
def fragile_parse(text: str) -> dict:
    intent = ""
    urgency = ""
    needs_reply = False
    summary = ""
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("intent:"):
            intent = line.split(":", 1)[1].strip()
        elif lower.startswith("urgency:"):
            urgency = line.split(":", 1)[1].strip()
        elif lower.startswith("needs reply:"):
            value = line.split(":", 1)[1].strip().lower()
            needs_reply = value == "yes"
        elif lower.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
    return {
        "intent": intent,
        "urgency": urgency,
        "needs_reply": needs_reply,
        "summary": summary,
    }

samples = {
    "plain": """Intent: Invoice
Urgency: High
Needs Reply: Yes
Summary: Please review.""",
    "markdown_bold": """**Intent:** Invoice
**Urgency:** High
**Needs Reply:** Yes
**Summary:** Please review.""",
    "period_after_yes": """Intent: Invoice
Urgency: Low
Needs Reply: Yes.
Summary: Please review.""",
}

for name, sample in samples.items():
    print(f"\n{name}:")
    print("Fragile parser:", fragile_parse(sample))

print("\nHardened structured-data validation:")
json_reply = '''```json
{"intent":"Invoice","urgency":"High","needs_reply":"Yes.","summary":"Please review."}
```'''
parsed = validate_classification(_extract_json(json_reply))
print(parsed)
print("\nTeaching point: the original parser can fail silently; structured data plus validation is safer.")
