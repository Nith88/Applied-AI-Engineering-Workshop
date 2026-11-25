"""A tiny educational graph runner. It is not LangGraph itself."""

END = "__END__"

def run_tiny_graph(state, nodes, edges, start):
    current = start
    while current != END:
        node = nodes[current]
        updates = node(state) or {}
        state = {**state, **updates}  # merge node updates into shared state
        next_step = edges[current]
        current = next_step(state) if callable(next_step) else next_step
    return state


def classify(state):
    subject = state["subject"].lower()
    if "urgent" in subject:
        return {"urgency": "High", "needs_reply": True}
    if "question" in subject:
        return {"urgency": "Medium", "needs_reply": True}
    return {"urgency": "Low", "needs_reply": False}


def decide(state):
    if state["urgency"] == "High":
        return {"action": "flag_for_review"}
    if state["needs_reply"]:
        return {"action": "draft_reply"}
    return {"action": "no_action"}


def draft(state):
    print("Drafting a reply for:", state["subject"])
    return {"draft_reply": "Thank you for your email. I will review it."}


def flag(state):
    print("Human review required:", state["subject"])
    return {}


def choose_outcome(state):
    return {
        "draft_reply": "draft",
        "flag_for_review": "flag",
        "no_action": END,
    }[state["action"]]


nodes = {
    "classify": classify,
    "decide": decide,
    "draft": draft,
    "flag": flag,
}
edges = {
    "classify": "decide",
    "decide": choose_outcome,
    "draft": END,
    "flag": END,
}

if __name__ == "__main__":
    for subject in [
        "URGENT: server down",
        "Question about tomorrow's class",
        "Monthly receipt",
    ]:
        initial = {"subject": subject, "urgency": "", "needs_reply": False,
                   "action": "", "draft_reply": ""}
        print("\nRUN:", subject)
        print("RESULT:", run_tiny_graph(initial, nodes, edges, "classify"))
