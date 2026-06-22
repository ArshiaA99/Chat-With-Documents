import uuid
import logging
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

logger = logging.getLogger(__name__)

def get_chroma_client():
    return chromadb.PersistentClient(path='./mydb')

def get_collection():
    client = get_chroma_client()
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name='documents',
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

def add_uploaded_file_to_index(uploaded_file) -> str:
    collection = get_collection()
    file_name = uploaded_file.filename
    
    # Defensive seek reset for upstream stream reads
    uploaded_file.file.seek(0)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = []
    document_id = str(uuid.uuid4())[:8]

    try:
        if file_name.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(uploaded_file.file)
            for page_idx, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_chunks = text_splitter.create_documents(
                        texts=[page_text],
                        metadatas=[{
                            "source": file_name, 
                            "page": page_idx,
                            "document_id": document_id
                        }]
                    )
                    chunks.extend(page_chunks)
                    
        elif file_name.endswith('.txt'):
            content = uploaded_file.file.read().decode("utf-8")
            if content.strip():
                chunks = text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{"source": file_name, "document_id": document_id}]
                )
    except Exception:
        logger.exception(f"Failed to parse incoming file stream for: {file_name}")
        return "❌ Error parsing file layer contents."

    if not chunks:
        return f"⚠️ Could not extract text contents from {file_name}."

    chroma_documents = [doc.page_content for doc in chunks]
    chroma_metadatas = []
    chroma_ids = []

    for idx, doc_chunk in enumerate(chunks):
        chunk_meta = {**doc_chunk.metadata, "chunk_index": idx}
        chroma_metadatas.append(chunk_meta)
        chroma_ids.append(f"{file_name}_{document_id}_ch_{idx}")

    collection.upsert(
        documents=chroma_documents,
        metadatas=chroma_metadatas,
        ids=chroma_ids
    )
    return f"✅ Successfully indexed '{file_name}' ({len(chroma_documents)} chunks)."