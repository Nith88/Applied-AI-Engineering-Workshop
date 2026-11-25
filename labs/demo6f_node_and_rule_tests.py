"""Small tests for deterministic workflow logic; no Gemini or Gmail needed."""

from workflow_core import decide_action, validate_classification

def test_decision_rules():
    assert decide_action({"urgency": "High", "needs_reply": True})["action"] == "flag_for_review"
    assert decide_action({"urgency": "High", "needs_reply": False})["action"] == "flag_for_review"
    assert decide_action({"urgency": "Medium", "needs_reply": True})["action"] == "draft_reply"
    assert decide_action({"urgency": "Low", "needs_reply": False})["action"] == "no_action"

def test_validation():
    valid = validate_classification({
        "intent": "Question", "urgency": "medium",
        "needs_reply": "Yes.", "summary": "Needs response"
    })
    assert valid["urgency"] == "Medium"
    assert valid["needs_reply"] is True

if __name__ == "__main__":
    test_decision_rules()
    test_validation()
    print("All tests passed.")
