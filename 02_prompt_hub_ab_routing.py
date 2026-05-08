"""
Step 2 — Prompt Hub & A/B Routing
===================================
Push two prompt versions to LangSmith Prompt Hub, pull them back,
and route queries deterministically (MD5 hash). Adds >= 50 more traces.
"""

import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from qa_pairs import QA_PAIRS

LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
PROMPT_V1_NAME = "my-rag-prompt-v1"
PROMPT_V2_NAME = "my-rag-prompt-v2"

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

# ---- Define two prompt templates -------------------------------------------
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a well-structured answer using ONLY the provided context.\n\n"
    "Instructions:\n"
    "1. Identify the key facts in the context that answer the question.\n"
    "2. Present those facts in a clear, organized manner (2-4 sentences).\n"
    "3. Do NOT add information beyond what the context provides.\n"
    "4. If the context lacks sufficient information, state this explicitly.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

LOCAL_PROMPTS = {
    PROMPT_V1_NAME: PROMPT_V1,
    PROMPT_V2_NAME: PROMPT_V2,
}


# ---- Push prompts to LangSmith Prompt Hub ----------------------------------
def push_prompts_to_hub(client):
    """Upload both prompt versions to LangSmith Prompt Hub."""
    for name, template, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 — concise answers (2-4 sentences)"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 — structured expert answers (3-5 sentences)"),
    ]:
        try:
            url = client.push_prompt(name, object=template, description=desc)
            print(f"✅ Pushed '{name}' → {url}")
        except Exception as e:
            print(f"⚠️  {name}: {e}")


# ---- Pull prompts from Prompt Hub ------------------------------------------
def pull_prompts_from_hub(client):
    """Download both prompt versions from Hub. Falls back to local on error."""
    prompts = {}
    for name in [PROMPT_V1_NAME, PROMPT_V2_NAME]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"↓ Pulled '{name}' from Hub")
        except Exception:
            prompts[name] = LOCAL_PROMPTS[name]
            print(f"ℹ️  Using local fallback for '{name}'")
    return prompts


# ---- A/B routing — deterministic hash --------------------------------------
def get_prompt_version(request_id: str) -> str:
    """Route deterministically via MD5 hash: even -> V1, odd -> V2."""
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if h % 2 == 0 else PROMPT_V2_NAME


# ---- Build vectorstore ------------------------------------------------------
def build_vectorstore():
    """Load knowledge base, split into chunks, and index with FAISS."""
    text = Path(__file__).parent / "data" / "knowledge_base.txt"
    content = text.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)
    print(f"Split into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, embeddings)


# ---- Traced A/B query function ---------------------------------------------
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm_instance, prompt, question: str, version: str) -> dict:
    """Run RAG chain with a specific prompt version. Returns result dict with version tag."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm_instance | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


# ---- Main ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    counts = {"v1": 0, "v2": 0}

    for i, qa in enumerate(QA_PAIRS):
        question = qa["question"]
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        counts[version_tag] += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:60]}")

    print(f"\nRouting summary: V1={counts['v1']}, V2={counts['v2']}")
    print(f"✅ 50 traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")


if __name__ == "__main__":
    main()
