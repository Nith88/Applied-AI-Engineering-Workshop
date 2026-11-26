
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------
# 1. INITIALIZATION
# --------------------------------------------------

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# --------------------------------------------------
# 2. DEFINE AGENT STATE
# --------------------------------------------------

class EmailState(TypedDict, total=False):
    email_text: str
    classification: str
    action: str
    draft: str


# --------------------------------------------------
# 3. EMAIL CLASSIFIER
# --------------------------------------------------

classifier = (
    ChatPromptTemplate.from_template(
        """
        Classify this email.

        Return:
        1. Intent: security_alert, meeting_request,
           newsletter, resource_request, other
        2. Urgency: low, medium, high
        3. Needs_reply: true or false
        4. Summary: one sentence

        Email:
        {email_text}
        """
    )
    | llm
)


def classify_email(state: EmailState):

    result = classifier.invoke({
        "email_text": state["email_text"]
    })

    # Convert AIMessage content into text
    classification_text = result.text

    print("\n[CLASSIFIER] Email classified.")

    return {
        "classification": classification_text
    }


# --------------------------------------------------
# 4. CONDITIONAL ROUTING
# --------------------------------------------------

def route_email(
    state: EmailState
) -> Literal["flag", "draft", "ignore"]:

    text = state.get("classification", "").lower()

    print("[ROUTER] Evaluating classification...")

    if "security_alert" in text or "urgency: high" in text:
        print("[ROUTER] Routing to human review.")
        return "flag"

    if "meeting_request" in text or "resource_request" in text:
        print("[ROUTER] Routing to reply drafting.")
        return "draft"

    print("[ROUTER] No action required.")
    return "ignore"


# --------------------------------------------------
# 5. ACTION NODES
# --------------------------------------------------

def flag_for_review(state: EmailState):

    print("[ACTION] Flagging email for human review.")

    return {
        "action": "Flag for human review."
    }


def draft_reply(state: EmailState):

    print("[ACTION] Generating email reply.")

    prompt = """
    Draft a polite and professional reply to this email.

    Do NOT send the email.
    Only prepare a draft for human review.

    Email:
    {email_text}
    """

    response = llm.invoke(
        prompt.format(email_text=state["email_text"])
    )

    return {
        "action": "Draft prepared for review.",
        "draft": response.text
    }


def ignore_newsletter(state: EmailState):

    print("[ACTION] Leaving email unchanged.")

    return {
        "action": "No reply needed; leave unchanged."
    }


# --------------------------------------------------
# 6. BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

graph_builder = StateGraph(EmailState)

# Register nodes
graph_builder.add_node("classify", classify_email)
graph_builder.add_node("flag", flag_for_review)
graph_builder.add_node("draft", draft_reply)
graph_builder.add_node("ignore", ignore_newsletter)

# Starting point
graph_builder.add_edge(START, "classify")

# Conditional routing
graph_builder.add_conditional_edges(
    "classify",
    route_email,
    {
        "flag": "flag",
        "draft": "draft",
        "ignore": "ignore"
    }
)

# Connect action nodes to END
for node in ["flag", "draft", "ignore"]:
    graph_builder.add_edge(node, END)

# Compile graph
graph = graph_builder.compile()


# --------------------------------------------------
# 7. TEST EMAILS
# --------------------------------------------------

examples = [
    "Your account was accessed from a new location. Verify immediately.",

    "Can we move our project meeting to Thursday afternoon?",

    "Weekly newsletter: five AI links you may enjoy."
]


# --------------------------------------------------
# 8. EXECUTE AGENT
# --------------------------------------------------

for email in examples:

    print("\n" + "=" * 60)
    print("INPUT EMAIL:", email)

    result = graph.invoke({
        "email_text": email
    })

    print("\nCLASSIFICATION:")
    print(result.get("classification"))

    print("\nFINAL ACTION:")
    print(result.get("action"))

    if result.get("draft"):
        print("\nDRAFT (NOT SENT):")
        print(result["draft"])

    print("=" * 60)
