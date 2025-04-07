from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from preprocessing.preprocessing import ResolveReferences
import httpx
import toml
import os
import json

CONFIG = toml.load("config.toml")
N = CONFIG["client"]["n"]
URL = CONFIG["client"]["url"]
GEMINI_CONFIG = CONFIG["gemini"]
SPLITTER_CONFIG = CONFIG["splitter"]
API_KEYS = toml.load("RAG/API_KEYS.toml")

# saving API KEYS in the environment
for key, item in API_KEYS.items():
    os.environ[key] = item

# initializing splitter
splitter = CharacterTextSplitter(**SPLITTER_CONFIG)

def save(documents:list[str], filenames:list[str]) -> str:
    response = httpx.post(
        url=URL+"/save",
        json={"documents":documents, "filenames":filenames}
    )
    result = json.loads(response.content)
    if result["response"] == "success":
        return "success"
    return result["exception"]

def reset() -> str:
    response = httpx.get(
        url=URL+"/reset", 
    )
    result = json.loads(response.content)
    if result["response"] == "success":
        return "success"
    return result["exception"]

def UpdateDatabase():
    if reset() == "success":
        for root, dirs, files in os.walk("./Files"):
            for filename in files:
                filepath = os.path.join(root, filename)
                with open(filepath, "r") as file:
                    text = open(filepath, "r").read()
                text = ResolveReferences(text) # resolve coreferences
                documents = splitter.split_text(text)
                response = save(documents, [filepath]*len(documents))
                if response != "success":
                    reset()
                    return "failure while saving data in the database. Auto resetting the database again"
        return "success"
    return "failure while resetting database"

def retrieve(query:str, n:int) -> list[str]|str:
    response = httpx.post(
        url=URL+"/retrieve", 
        json={"query":query, "n":n}
    )
    result = json.loads(response.content)
    if result["response"] == "success":
        print(result["documents"])
        return result["documents"] # list[str]
    return result["exception"] # str

RAGPrompt = ChatPromptTemplate.from_messages([
    ("system", "{context}\n\n Answer the below query based on the above question"),
    ("human", "query: {query}")
])

LLM = ChatGoogleGenerativeAI(**GEMINI_CONFIG)
LLMChain = RAGPrompt | LLM

class State(TypedDict):
    query: str
    context: list[str]
    answer: str

def Retrieve(State:State):
    context = retrieve(State["query"], n=N)[0]
    return {"query":State["query"], "context": context, "answer":State["answer"]}

def LLMQuery(State: State):
    context = "\n".join(State["context"])
    answer = LLMChain.invoke({
        "context": context,
        "query": State["query"]
    }).content
    return {"query": State["query"], "context": State["context"], "answer":answer}

builder = StateGraph(State)

builder.add_node("retriever", Retrieve)
builder.add_node("llm", LLMQuery)

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "llm")
builder.add_edge("llm", END)

graph = builder.compile()