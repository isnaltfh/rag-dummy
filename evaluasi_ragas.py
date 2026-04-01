# ── Import Library ──────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("EVALUASI RAG MENGGUNAKAN RAGAS")
print("Metrik: Faithfulness & Context Recall")
print("Judge : Llama 3.2 via Ollama (Self-Hosted)")
print("=" * 55)

# ── Step 1: Load ChromaDB ────────────────────────────────────
print("\nMemuat ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("   OK: ChromaDB berhasil dimuat!")

# ── Step 2: Setup RAG Pipeline ──────────────────────────────
print("\nMenghubungkan ke Llama 3.2 via Ollama...")
llm = OllamaLLM(model="llama3.2", timeout=600)

prompt_template = PromptTemplate.from_template("""
Answer the following question based only on the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
print("   OK: Pipeline RAG siap!\n")

# ── Step 3: Setup RAGAS dengan Ollama sebagai Judge ──────────
print("Menyiapkan RAGAS Judge menggunakan Ollama...")
ollama_chat = ChatOllama(model="llama3.2", timeout=600, num_predict=512)
ragas_llm = LangchainLLMWrapper(ollama_chat)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

faithfulness.llm = ragas_llm
faithfulness.embeddings = ragas_embeddings
context_recall.llm = ragas_llm
context_recall.embeddings = ragas_embeddings
print("   OK: RAGAS Judge siap!\n")

# ── Step 4: Dataset Evaluasi (1 pertanyaan dulu untuk test) ──
# Pakai bahasa Inggris agar Llama 3.2 lebih akurat mengevaluasi
eval_dataset = [
    {
        "question": "What is faithfulness in RAG evaluation?",
        "ground_truth": "Faithfulness measures whether the claims made in the generated answer can be logically inferred from the retrieved context, ensuring the answer does not contain hallucinated information."
    },
]

# ── Step 5: Jalankan RAG ─────────────────────────────────────
print("Menjalankan RAG untuk setiap pertanyaan...")
questions = []
answers = []
contexts = []
ground_truths = []

for i, item in enumerate(eval_dataset):
    print(f"\n[{i+1}/{len(eval_dataset)}] {item['question']}")
    answer = rag_chain.invoke(item["question"])
    source_docs = retriever.invoke(item["question"])
    context_texts = [doc.page_content for doc in source_docs]

    questions.append(item["question"])
    answers.append(answer)
    ground_truths.append(item["ground_truth"])
    contexts.append(context_texts)
    print(f"   Jawaban: {answer[:100]}...")
    print(f"   Konteks ditemukan: {len(context_texts)} chunks")

# ── Step 6: Evaluasi RAGAS ───────────────────────────────────
print("\nMenyiapkan dataset RAGAS...")
ragas_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

print("\nMenjalankan evaluasi RAGAS...")
print("(Harap tunggu, proses ini membutuhkan waktu ~5 menit...)\n")

results = evaluate(
    dataset=ragas_dataset,
    metrics=[faithfulness, context_recall],
    raise_exceptions=False,
)

# ── Step 7: Tampilkan Hasil ──────────────────────────────────
print("\n" + "=" * 55)
print("HASIL EVALUASI RAGAS")
print("=" * 55)

df = results.to_pandas()
print(f"Kolom tersedia: {df.columns.tolist()}\n")

# Mapping nama kolom baru RAGAS
col_map = {
    "user_input": "question",
    "response": "answer",
    "retrieved_contexts": "contexts",
    "reference": "ground_truth"
}

print("=" * 55)
if "faithfulness" in df.columns:
    val = df["faithfulness"].iloc[0]
    if str(val) != "nan":
        print(f"✅ FAITHFULNESS   : {val:.4f} ({val*100:.1f}%)")
    else:
        print("⚠️  FAITHFULNESS   : Timeout — coba jalankan ulang")
else:
    print("⚠️  FAITHFULNESS   : Kolom tidak tersedia")

if "context_recall" in df.columns:
    val = df["context_recall"].iloc[0]
    if str(val) != "nan":
        print(f"✅ CONTEXT RECALL  : {val:.4f} ({val*100:.1f}%)")
    else:
        print("⚠️  CONTEXT RECALL  : Timeout — coba jalankan ulang")
else:
    print("⚠️  CONTEXT RECALL  : Kolom tidak tersedia")

print("=" * 55)
print("\nEvaluasi selesai!")
print("\nCATATAN: Jika masih NaN, evaluasi resmi akan dilakukan")
print("di server VMware PLN dengan hardware yang lebih powerful.")
