import os
import tempfile
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from PIL import Image

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "rag_collection"
SUPPORTED_EXTENSIONS = {"pdf", "doc", "docx", "ppt", "pptx", "csv", "jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 50

_vectorstore: Chroma | None = None


def _get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAIEmbeddings(api_key=api_key)


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)


def wipe_vectorstore():
    """Destroy the in-memory vectorstore completely. No disk I/O needed."""
    global _vectorstore
    if _vectorstore is not None:
        try:
            _vectorstore._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _vectorstore = None
    print("[INFO] Vectorstore wiped.")


def vectorstore_has_documents() -> bool:
    global _vectorstore
    if _vectorstore is None:
        return False
    try:
        return _vectorstore._collection.count() > 0
    except Exception:
        return False
    

def load_image_as_document(file_path: str, filename: str) -> List[Document]:
    try:
        import pytesseract
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": filename})]
    except ImportError:
        print("[WARN] pytesseract not installed – skipping image OCR")
        return []
    except Exception as e:
        print(f"[WARN] OCR failed for {filename}: {e}")
        return []


def load_file(file_path: str, filename: str) -> List[Document]:
    ext = filename.lower().rsplit(".", 1)[-1]
    try:
        if ext == "pdf":
            return PyPDFLoader(file_path).load()
        elif ext in {"doc", "docx"}:
            return Docx2txtLoader(file_path).load()
        elif ext in {"ppt", "pptx"}:
            return UnstructuredPowerPointLoader(file_path).load()
        elif ext == "csv":
            return CSVLoader(file_path).load()
        elif ext in {"jpg", "jpeg", "png"}:
            return load_image_as_document(file_path, filename)
        else:
            return []
    except Exception as e:
        print(f"[WARN] Failed to load {filename}: {e}")
        return []


def sanitize_documents(docs: List[Document]) -> List[Document]:
    clean = []
    for doc in docs:
        if not doc.page_content:
            continue
        if not isinstance(doc.page_content, str):
            continue
        if not doc.page_content.strip():
            continue
        if doc.metadata is None:
            doc.metadata = {}
        clean.append(doc)
    return clean


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global _vectorstore

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for f in files:
        ext = f.filename.lower().rsplit(".", 1)[-1] if "." in f.filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{ext} ({f.filename})"
            )

    all_documents: List[Document] = []
    failed_files: List[str] = []
    temp_dir = tempfile.mkdtemp()

    try:
        for uploaded_file in files:
            temp_path = os.path.join(temp_dir, uploaded_file.filename)
            content = await uploaded_file.read()

            if len(content) / (1024 * 1024) > MAX_FILE_SIZE_MB:
                failed_files.append(f"{uploaded_file.filename} (exceeds {MAX_FILE_SIZE_MB}MB)")
                continue

            with open(temp_path, "wb") as fh:
                fh.write(content)

            docs = load_file(temp_path, uploaded_file.filename)
            if docs:
                all_documents.extend(docs)
            else:
                failed_files.append(uploaded_file.filename)

        if not all_documents:
            raise HTTPException(status_code=422, detail="No content could be extracted.")

        all_documents = sanitize_documents(all_documents)

        if not all_documents:
            raise HTTPException(status_code=422, detail="All pages were blank or unreadable.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_documents)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

        if not chunks:
            raise HTTPException(status_code=422, detail="No usable text chunks after processing.")

    
        wipe_vectorstore()

       
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=_get_embeddings(),
            collection_name=COLLECTION_NAME,
        )

        count = _vectorstore._collection.count()
        print(f"[INFO] Vectorstore ready. Total chunks: {count}")

        result = {
            "status": "ready",
            "chunks": len(chunks),
            "files_processed": len(files) - len(failed_files),
            "files_total": len(files),
        }
        if failed_files:
            result["warnings"] = f"Failed to load: {', '.join(failed_files)}"

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global _vectorstore

    if not vectorstore_has_documents():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retriever = _vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 10},
        )
        docs = retriever.invoke(question)

        if not docs:
            return {"answer": "I couldn't find relevant information in the uploaded documents."}

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = PromptTemplate.from_template(
    """You are a precise document assistant. A user has uploaded documents and is asking questions about them.

RULES:
- Answer using ONLY the information present in the context below.
- If the question has multiple parts, address each part separately.
- If a specific piece of information is not found in the context, say "Not found in document" for that part only.
- Never fabricate, assume, or infer information not explicitly stated.
- Be concise but complete. Do not repeat the question back.
- If the answer is a list, format it clearly.

Context:
{context}

Question: {question}

Answer:"""
        )

        chain = prompt | get_llm() | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
async def reset_vectorstore():
    wipe_vectorstore()
    return {"status": "cleared"}


@app.get("/health")
async def health():
    ready = vectorstore_has_documents()
    return {"status": "healthy", "vectorstore_ready": ready}