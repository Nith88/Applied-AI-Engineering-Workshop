import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)
parser = StrOutputParser()

# 1. Simple chain.
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to a 10-year-old with one analogy and one example."
)
simple_chain = prompt | llm | parser
print(simple_chain.invoke({"topic": "machine learning"}))

# 2. Sequential chain: each stage receives the previous output.
question_chain = (
    ChatPromptTemplate.from_template("Create three research questions about {topic}")
    | llm | parser
)
answer_chain = (
    ChatPromptTemplate.from_template("Answer these questions concisely:\\n{questions}")
    | llm | parser
)
summary_chain = (
    ChatPromptTemplate.from_template("Summarise these answers in five bullets:\\n{answers}")
    | llm | parser
)
questions = question_chain.invoke({"topic": "AI in education"})
answers = answer_chain.invoke({"questions": questions})
summary = summary_chain.invoke({"answers": answers})
print(summary)

# 3. Multi-input chain.
lesson_chain = (
    ChatPromptTemplate.from_template(
        "Create a 30-minute lesson plan. Topic: {topic}. "
        "Level: {level}. Outcome: {outcome}. Include activity and assessment."
    ) | llm | parser
)
print(lesson_chain.invoke({
    "topic": "Java interfaces",
    "level": "first-year B.Tech",
    "outcome": "Students can explain and implement an interface"
}))

# 4. Batch processing.
glossary_chain = (
    ChatPromptTemplate.from_template("Define {term} simply and give one example.")
    | llm | parser
)
terms = [{"term": t} for t in ["Embedding", "Vector database", "RAG", "Agent"]]
for definition in glossary_chain.batch(terms):
    print(definition)
