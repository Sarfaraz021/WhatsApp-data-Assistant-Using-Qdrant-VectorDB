from langchain_community.document_loaders import TextLoader
from prompt import template
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
import streamlit as st
from dotenv import load_dotenv
import os
# import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Load environment variables
load_dotenv("var.env")
os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.getenv("LANGCHAIN_API_KEY")

# Set embeddings
embd = OpenAIEmbeddings(model="text-embedding-3-large")


# Load and split documents
@st.cache_resource
def load_and_process_documents():
    # Use os.path.join for cross-platform compatibility
    file_path = os.path.join(os.path.dirname(__file__), 'Data', 'liran.txt')

    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            docs = loader.load()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(
            f"Unable to load {file_path} with any of the attempted encodings")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=10000, chunk_overlap=2000
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    return vectorstore.as_retriever()


retriever = load_and_process_documents()
# Router


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore"] = Field(
        ...,
        description="Given a user question choose to route it to a vectorstore or use your own knowledge.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Routing prompt
system = """You are an expert at routing a user question to a vectorstore.
The vectorstore contains documents related to Alzheimer's and Dementia a leading voluntary health organization in Alzheimer's care, support and research.
Use the vectorstore for questions on related to this organization. Otherwise, use your own knowledge."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# Generate
prompt = PromptTemplate(
    input_variables=["context", "question", "history"],
    template=template
)

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Graph state


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    memory: ConversationSummaryMemory


def initialize_memory():
    return ConversationSummaryMemory(llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0))


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "memory": state["memory"]}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    memory = state["memory"]

    history = memory.load_memory_variables({})["history"]

    generation = rag_chain.invoke({
        "context": format_docs(documents),
        "question": question,
        "history": history
    })

    memory.save_context({"input": question}, {"output": generation})

    return {"documents": documents, "question": question, "generation": generation, "memory": memory}


def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    return "vectorstore" if source.datasource == "vectorstore" else "generate"


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Streamlit UI
st.title("AI Support Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = initialize_memory()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        inputs = {
            "question": prompt,
            "memory": st.session_state.memory
        }

        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    full_response = value["generation"]
                    message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.session_state.memory = value["memory"]

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})


# import sqlite3
# print(sqlite3.sqlite_version)
