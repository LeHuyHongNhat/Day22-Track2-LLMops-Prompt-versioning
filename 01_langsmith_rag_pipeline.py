"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
Build a RAG pipeline with FAISS vector search and LangSmith tracing.
Run all 50 questions → generates >= 50 LangSmith traces.
"""

import os
from pathlib import Path

# Load .env and set LangSmith env vars BEFORE importing LangChain
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import QA_PAIRS


# ---- LLM and Embeddings ----------------------------------------------------
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

embeddings = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


# ---- Build FAISS vector store ----------------------------------------------
def build_vectorstore():
    """Load knowledge base, split into chunks, and index with FAISS."""
    text = Path(__file__).parent / "data" / "knowledge_base.txt"
    content = text.read_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ---- RAG prompt template ---------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the context below to answer the question. If the context does not contain the answer, say: 'I don't have enough information.'\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ---- Build RAG chain (LCEL) ------------------------------------------------
def build_rag_chain(vectorstore):
    """Build LCEL RAG chain: retriever -> format_docs -> prompt -> LLM -> parser."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ---- Traced query function -------------------------------------------------
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Invoke RAG chain. @traceable sends input/output/latency to LangSmith."""
    return chain.invoke(question)


# ---- Main ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:80]}")
        print(f"       A: {answer[:120]}\n")

    print(f"✅ {len(QA_PAIRS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
