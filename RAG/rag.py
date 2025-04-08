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

def UpdateDatabase(resolve_refs=True):
    if reset() == "success":
        for root, dirs, files in os.walk("./Files"):
            for filename in files:
                filepath = os.path.join(root, filename)
                with open(filepath, "r") as file:
                    text = open(filepath, "r").read()
                if resolve_refs:
                    text = ResolveReferences(text) # resolve coreferences
                documents = splitter.split_text(text)
                response = save(documents, [filepath]*len(documents))
                if response != "success":
                    reset()
                    return "failure while saving data in the database. Auto resetting the database again"
        return "success"
    return "failure while resetting database"

def retrieve(query:str, n:int) -> list[str]:
    response = httpx.post(
        url=URL+"/retrieve", 
        json={"query":query, "n":n},
        timeout=60
    )
    result = json.loads(response.content)
    if result["response"] == "success":
        return result["documents"] # list[str]
    else:
        print(result["exception"])
        return []

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
    context = retrieve(
        query=State["query"], 
        n=N
    )
    print("retrieved context:")
    print(context)
    return {"query":State["query"], "context": context, "answer":State["answer"]}

def LLMQuery(State:State):
    context = "\n".join(State["context"])
    answer = LLMChain.invoke({
        "context": context,
        "query": State["query"]
    }).content
    return {"query": State["query"], "context": State["context"], "answer":answer}

builder_retriver = StateGraph(State)
builder_retriver.add_node("retriever", Retrieve)
builder_retriver.add_node("llm", LLMQuery)
builder_retriver.add_edge(START, "retriever")
builder_retriver.add_edge("retriever", "llm")
builder_retriver.add_edge("llm", END)
graph_with_retriever = builder_retriver.compile()


builder_direct = StateGraph(State)
builder_direct.add_node("llm", LLMQuery)
builder_direct.add_edge(START, "llm")
builder_direct.add_edge("llm", END)
graph_direct = builder_direct.compile()


