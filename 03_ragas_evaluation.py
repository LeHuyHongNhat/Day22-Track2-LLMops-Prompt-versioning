"""
Step 3 — RAGAS Evaluation
===========================
Run all 50 QA pairs through BOTH prompt versions, evaluate with 4 RAGAS metrics,
print comparison table, save report to data/ragas_report.json.

NOTE: This step takes ~20-30 minutes due to many LLM calls.
"""

import os
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from qa_pairs import QA_PAIRS

# ---- LLM and Embeddings (for RAG chain) ------------------------------------
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

# ---- Prompt templates (same as step 2) -------------------------------------
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ---- Build vectorstore -----------------------------------------------------
def build_vectorstore():
    """Load knowledge base, split into chunks, and index with FAISS."""
    text = Path(__file__).parent / "data" / "knowledge_base.txt"
    content = text.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)
    print(f"Split into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, embeddings)


# ---- Run RAG and capture outputs + contexts --------------------------------
def run_rag(retriever, llm_instance, prompt, question: str) -> dict:
    """Run a single RAG query. Returns answer and contexts as list[str]."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm_instance | StrOutputParser()).invoke(
        {"context": ctx_str, "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Run all 50 QA pairs through a prompt version and collect results."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, PROMPTS[prompt_version], qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        sys.stdout.write(f"\r  [{i:02d}/50] {qa['question'][:70]}")
        sys.stdout.flush()
    print()
    return results


# ---- Build RAGAS EvaluationDataset -----------------------------------------
def build_ragas_dataset(rag_results: list):
    """Convert RAG result dicts into a RAGAS EvaluationDataset."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ---- Run RAGAS evaluation --------------------------------------------------
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Evaluate RAG outputs with 4 RAGAS metrics. Returns dict of mean scores."""
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    llm_eval = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    emb_eval = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ---- Main ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # Comparison table
    print("\n" + "=" * 70)
    print("  V1 vs V2 Comparison")
    print("=" * 70)
    print(f"  {'Metric':30s}  {'V1':>8s}  {'V2':>8s}  Winner")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  ------")

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    for metric in metrics:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        if s1 > s2:
            winner = "← V1"
        elif s2 > s1:
            winner = "← V2"
        else:
            winner = "tie"
        print(f"  {metric:30s}: {s1:.4f}   {s2:.4f}   {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    print()
    if best_faith >= 0.8:
        print(f"✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # Save report
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "target_faithfulness": 0.8,
        "best_faithfulness": best_faith,
    }
    out_path = Path(__file__).parent / "data" / "ragas_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"💾 Saved data/ragas_report.json")


if __name__ == "__main__":
    main()
