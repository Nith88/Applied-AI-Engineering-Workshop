from datetime import datetime
from langchain_core.tools import tool

@tool
def current_datetime() -> str:
    """Return the computer's current local date and time."""
    return datetime.now().astimezone().isoformat(timespec="minutes")

@tool
def count_words(text: str) -> str:
    """Count words and sentence-ending marks in text."""
    words = len(text.split())
    sentences = sum(text.count(mark) for mark in ".!?")
    return f"Words: {words}; sentence-ending marks: {sentences}"

@tool
def make_study_plan(topic: str, days: int) -> str:
    """Create a simple day-wise study plan for a topic."""
    if not 1 <= days <= 30:
        return "Choose between 1 and 30 days."
    return "\\n".join(
        f"Day {day}: Study one subtopic of {topic} and practise questions."
        for day in range(1, days + 1)
    )

print(current_datetime.invoke({}))
print(count_words.invoke({"text": "RAG retrieves context. The model uses it to answer."}))
print(make_study_plan.invoke({"topic": "DBMS", "days": 3}))
