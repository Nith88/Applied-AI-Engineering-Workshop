import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


@tool
def current_datetime() -> str:
    """Return the computer's current local date and time as an ISO 8601 string."""
    return datetime.now().astimezone().isoformat(timespec="minutes")


@tool
def count_words(text: str) -> int:
    """Count whitespace-separated words in non-empty text."""
    if not text.strip():
        raise ValueError("text must contain at least one non-whitespace character")
    return len(text.split())


@tool
def make_study_plan(topic: str, days: int) -> str:
    """Create a simple day-by-day study plan for a topic, for 1 to 30 days."""
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 30:
        raise ValueError("days must be an integer between 1 and 30")

    topic = topic.strip()
    return "\n".join(
        f"Day {day}: Study {topic} and practise questions."
        for day in range(1, days + 1)
    )


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0
)

tools = [current_datetime, count_words, make_study_plan]
tools_by_name = {selected_tool.name: selected_tool for selected_tool in tools}

# bind_tools tells the model which tools it may request; it does not execute them.
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly assistant. Use conversation history when relevant. "
        "Call an available tool when it is useful for the request.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


def call_model_with_tools(prompt_value: PromptValue) -> AIMessage:
    # Tool calls and their results are sent back to the model until it answers.
    messages: list[BaseMessage] = prompt_value.to_messages()
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError("The chat model must return an AIMessage.")
        if not response.tool_calls:
            return response

        messages.append(response)
        for tool_call in response.tool_calls:
            selected_tool = tools_by_name.get(tool_call["name"])
            if selected_tool is None:
                raise ValueError(f"The model requested an unknown tool: {tool_call['name']}")

            print(f"Calling tool: {selected_tool.name}")
            result = selected_tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    # Do not let an unexpected sequence of tool requests loop forever.
    response = llm_with_tools.invoke(messages)
    if not isinstance(response, AIMessage):
        raise TypeError("The chat model must return an AIMessage.")
    if response.tool_calls:
        raise RuntimeError("The assistant exceeded five tool-call rounds.")
    return response


chain = prompt | RunnableLambda(call_model_with_tools)

histories: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in histories:
        histories[session_id] = ChatMessageHistory()
    return histories[session_id]


chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

student_1 = {"configurable": {"session_id": "student-01"}}
student_2 = {"configurable": {"session_id": "student-02"}}

# The same session ID reuses history; another ID starts a separate conversation.
print("Student 1:", chat.invoke({"input": "My name is Nitheesh."}, config=student_1).content)
print(
    "Student 2:",
    chat.invoke(
        {"input": "My name is Asha. Use make_study_plan for a 3-day DBMS plan."},
        config=student_2,
    ).content,
)
print(
    "Student 1:",
    chat.invoke(
        {"input": "What is my name? Use current_datetime to tell me the local date and time."},
        config=student_1,
    ).content,
)
print(
    "Student 2:",
    chat.invoke(
        {"input": "What is my name? Count the words in: RAG retrieves evidence."},
        config=student_2,
    ).content,
)

# These direct calls test each tool without asking the model to choose one.
current_datetime_result = current_datetime.invoke({})
assert datetime.fromisoformat(current_datetime_result).utcoffset() is not None
print("Current date and time:", current_datetime_result)

word_count = count_words.invoke({"text": "RAG retrieves evidence. Models generate answers!"})
assert word_count == 6
print("Word count:", word_count)

study_plan = make_study_plan.invoke({"topic": "DBMS", "days": 3})
assert len(study_plan.splitlines()) == 3
print(study_plan)

for invalid_input in ("", "   "):
    try:
        count_words.invoke({"text": invalid_input})
    except ValueError:
        pass
    else:
        raise AssertionError("count_words should reject blank text")

for invalid_input in ({"topic": "DBMS", "days": 0}, {"topic": " ", "days": 3}):
    try:
        make_study_plan.invoke(invalid_input)
    except ValueError:
        pass
    else:
        raise AssertionError("make_study_plan should reject invalid inputs")

print("Tool .invoke() checks passed.")