import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are friendly. Use the conversation history when relevant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()
histories = {}

def get_history(session_id):
    if session_id not in histories:
        histories[session_id] = ChatMessageHistory()
    return histories[session_id]

chat = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="question",
    history_messages_key="history"
)
config = {"configurable": {"session_id": "student-01"}}
print(chat.invoke({"question": "My name is Nitheesh."}, config=config))
print(chat.invoke({"question": "What is my name?"}, config=config))
