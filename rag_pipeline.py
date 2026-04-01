# ── Import Library ──────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ── Step 1: Load semua PDF dari folder docs ─────────────────
print("Loading dokumen PDF...")
docs_folder = "docs"
all_documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(docs_folder, filename)
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"   OK: {filename} ({len(documents)} halaman)")

print(f"\nTotal halaman dimuat: {len(all_documents)}")

# ── Step 2: Chunking dokumen ────────────────────────────────
print("\nMelakukan chunking dokumen...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(all_documents)
print(f"   OK: Total chunks: {len(chunks)}")

# ── Step 3: Embedding dan simpan ke ChromaDB ────────────────
print("\nMembuat embedding dan menyimpan ke ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("   OK: ChromaDB berhasil dibuat!")

# ── Step 4: Setup RAG Pipeline ──────────────────────────────
print("\nMenghubungkan ke Llama 3.2 via Ollama...")
llm = OllamaLLM(model="llama3.2")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = PromptTemplate.from_template("""
Jawab pertanyaan berikut berdasarkan konteks yang diberikan saja.
Jika jawabannya tidak ada dalam konteks, katakan "Saya tidak tahu".

Konteks:
{context}

Pertanyaan: {question}

Jawaban:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

print("   OK: Pipeline RAG siap digunakan!\n")

# ── Step 5: Tanya Jawab ─────────────────────────────────────
print("=" * 50)
print("SISTEM TANYA JAWAB RAG SIAP")
print("Ketik 'exit' untuk keluar")
print("=" * 50)

while True:
    question = input("\nPertanyaan: ")
    if question.lower() == "exit":
        print("Keluar dari sistem. Sampai jumpa!")
        break

    print("\nSedang mencari jawaban...")
    
    # Ambil jawaban
    answer = rag_chain.invoke(question)
    
    # Ambil source dokumen
    source_docs = retriever.invoke(question)
    
    print(f"\nJawaban:\n{answer}")
    print("\nSumber dokumen:")
    for doc in source_docs:
        print(f"   - {doc.metadata.get('source', 'Unknown')} "
              f"(halaman {doc.metadata.get('page', '?')+1})")
