import logging
from typing import List
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from vectorstore import get_collection, get_chroma_client, add_uploaded_file_to_index
from rag import retrieve_context
from llm import ask_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise RAG Dashboard")
collection = get_collection()

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat With Documents</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans antialiased flex flex-col">

    <header class="border-b border-slate-800 bg-slate-950/60 backdrop-blur sticky top-0 z-20 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="p-2 bg-indigo-600 rounded-lg text-white">
                <i data-lucide="cpu" class="w-6 h-6"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">Chat With Documents</h1>
                <p class="text-xs text-slate-400">FastAPI Architecture • ChromaDB Vector Storage • Llama 3.3-70B</p>
            </div>
        </div>
        <div class="text-xs text-slate-400 bg-slate-800/60 px-3 py-1.5 rounded-full border border-slate-700 flex items-center space-x-2">
            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>ChromaDB Active</span>
        </div>
    </header>

    <div class="flex-1 flex flex-col md:flex-row max-w-[1600px] w-full mx-auto overflow-hidden">
        
        <aside class="w-full md:w-80 bg-slate-950 border-b md:border-b-0 md:border-r border-slate-800 p-6 flex flex-col justify-between shrink-0">
            <div class="space-y-6">
                <div>
                    <div class="flex items-center space-x-2 text-indigo-400 font-semibold uppercase tracking-wider text-xs mb-3">
                        <i data-lucide="download-cloud" class="w-4 h-4"></i>
                        <span>📥 Document Management</span>
                    </div>
                    <form id="upload-form" class="space-y-3">
                        <label class="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-4 cursor-pointer transition-colors bg-slate-900/50">
                            <i data-lucide="file-plus" class="w-7 h-7 text-slate-500 mb-2"></i>
                            <span class="text-xs font-medium text-slate-300">Upload reference documents</span>
                            <input type="file" id="file-input" name="files" accept=".pdf,.txt" multiple class="hidden" onchange="updateFileList()">
                        </label>
                        <div id="selected-files-list" class="text-[11px] text-slate-400 space-y-1 font-mono max-h-16 overflow-y-auto"></div>
                        <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98] transition-all text-white font-medium text-xs py-2.5 px-4 rounded-xl flex items-center justify-center space-x-2 shadow-md">
                            <span>Index Documents</span>
                        </button>
                    </form>
                </div>

                <div class="border-t border-slate-800 pt-4">
                    <div class="flex items-center space-x-2 text-cyan-400 font-semibold uppercase tracking-wider text-xs mb-3">
                        <i data-lucide="bar-chart-2" class="w-4 h-4"></i>
                        <span>📊 Database Stats</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 bg-slate-900/60 p-3 rounded-xl border border-slate-800/80">
                        <div class="bg-slate-950 p-2 rounded-lg border border-slate-800/40">
                            <p class="text-[10px] text-slate-400 uppercase font-medium">Documents</p>
                            <p id="stat-docs" class="text-lg font-bold text-slate-100">0</p>
                        </div>
                        <div class="bg-slate-950 p-2 rounded-lg border border-slate-800/40">
                            <p class="text-[10px] text-slate-400 uppercase font-medium">Total Chunks</p>
                            <p id="stat-chunks" class="text-lg font-bold text-slate-100">0</p>
                        </div>
                        <div class="col-span-2 bg-slate-950 p-2 rounded-lg border border-slate-800/40 mt-1">
                            <p class="text-[10px] text-slate-400 uppercase font-medium">Embedding Engine</p>
                            <p id="stat-model" class="text-[11px] font-mono font-bold text-indigo-300 truncate mt-0.5">Fetching...</p>
                        </div>
                    </div>
                </div>

                <div id="upload-status-container" class="space-y-2 max-h-32 overflow-y-auto"></div>
            </div>

            <div class="border-t border-slate-800 pt-4 mt-6">
                <button id="clear-db-btn" class="w-full bg-rose-950/40 hover:bg-rose-900/40 border border-rose-800/40 text-rose-300 hover:text-rose-200 transition-colors font-medium text-xs py-2.5 px-4 rounded-xl flex items-center justify-center space-x-2">
                    <i data-lucide="trash-2" class="w-4 h-4"></i>
                    <span>Wipe Vector Index</span>
                </button>
            </div>
        </aside>

        <main class="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 overflow-hidden">
            
            <section class="lg:col-span-7 flex flex-col bg-slate-950/50 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden min-h-[500px]">
                <div class="px-6 py-4 border-b border-slate-800 bg-slate-950 flex items-center space-x-2">
                    <i data-lucide="message-square" class="text-indigo-400 w-5 h-5"></i>
                    <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">💬 AI Knowledge Terminal</h2>
                </div>

                <div id="chat-window" class="flex-1 p-6 overflow-y-auto space-y-4 max-h-[520px]">
                    <div class="flex items-start gap-3" id="welcome-message">
                        <div class="w-8 h-8 rounded-full bg-indigo-950 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
                            <i data-lucide="bot" class="w-4 h-4"></i>
                        </div>
                        <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none px-4 py-3 max-w-[85%] text-sm text-slate-300 leading-relaxed">
                            Hello! I am your isolated RAG assistant. Query me regarding your loaded data, and I will generate an answer strictly limited to your ingested snippets.
                        </div>
                    </div>
                </div>

                <div id="chat-loading" class="hidden px-6 py-3 flex items-center space-x-2 text-xs text-indigo-400 border-t border-slate-900/50 bg-slate-900/20 font-mono">
                    <i data-lucide="loader" class="w-4 h-4 animate-spin"></i>
                    <span>Scanning system vectors & compiling context payload...</span>
                </div>

                <div class="p-4 bg-slate-950 border-t border-slate-800">
                    <form id="chat-form" class="flex gap-2">
                        <input type="text" id="chat-input" placeholder="Ask a question about your documents..." required autocomplete="off"
                            class="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all">
                        <button type="submit" class="bg-indigo-600 hover:bg-indigo-500 active:scale-95 transition-all text-white px-5 py-3 rounded-xl flex items-center gap-2 font-medium text-sm">
                            <span>Send</span>
                            <i data-lucide="send" class="w-4 h-4"></i>
                        </button>
                    </form>
                </div>
            </section>

            <section class="lg:col-span-5 flex flex-col gap-4">
                
                <div class="bg-slate-950/50 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col">
                    <div class="flex items-center gap-2 text-indigo-400 mb-3 text-xs font-semibold uppercase tracking-wider">
                        <i data-lucide="file-text" class="w-4 h-4"></i>
                        <span>📄 Citations & Vector Distance Scores</span>
                    </div>
                    <div id="inspector-source" class="text-xs font-mono bg-slate-900 p-3 rounded-xl border border-slate-800 text-slate-400 min-h-[65px] whitespace-pre-line leading-relaxed">
                        None
                    </div>
                </div>

                <div class="flex-1 bg-slate-950/50 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col">
                    <div class="flex items-center gap-2 text-cyan-400 mb-3 text-xs font-semibold uppercase tracking-wider">
                        <i data-lucide="database" class="w-4 h-4"></i>
                        <span>🗄️ Context Injection Payloads Sent to LLM</span>
                    </div>
                    <div class="flex-1 bg-slate-900 rounded-xl border border-slate-800 p-4 overflow-hidden flex flex-col min-h-[300px]">
                        <textarea id="inspector-context" readonly class="w-full h-full bg-transparent text-xs text-slate-400 font-mono resize-none focus:outline-none border-none leading-relaxed" 
                            placeholder="No query executed yet."></textarea>
                    </div>
                </div>

            </section>
        </main>
    </div>

    <script>
        lucide.createIcons();

        async function fetchDatabaseStats() {
            try {
                const res = await fetch('/stats');
                const stats = await res.json();
                document.getElementById('stat-docs').textContent = stats.documents_count || 0;
                document.getElementById('stat-chunks').textContent = stats.chunks_count || 0;
                document.getElementById('stat-model').textContent = stats.embedding_model || "None Loaded";
                const fileListContainer = document.getElementById('selected-files-list');
                if (stats.documents_list && stats.documents_list.length > 0) {
                    fileListContainer.innerHTML = '<p class="text-[10px] uppercase font-semibold text-indigo-400 mt-2 mb-1">Indexed Files:</p>';
                    stats.documents_list.forEach(doc => {
                        fileListContainer.innerHTML += `
                            <div class="flex items-center gap-1 text-[11px] text-slate-300 truncate bg-slate-900 p-1 rounded border border-slate-800 font-mono">
                                <i data-lucide="file" class="w-3 h-3 text-cyan-400 shrink-0"></i>${doc}
                            </div>`;
                    });
                    lucide.createIcons();
                } else {
                    fileListContainer.innerHTML = '<span class="text-slate-500 italic">No files indexed yet.</span>';
                }
            } catch (err) {
                console.error("Error updating statistics dashboard grid.", err);
            }
        }
        fetchDatabaseStats();

        function updateFileList() {
            const input = document.getElementById('file-input');
            const list = document.getElementById('selected-files-list');
            list.innerHTML = '';
            for(let file of input.files) {
                list.innerHTML += `<div class="flex items-center gap-1 text-[10px]"><i data-lucide="check" class="w-3 h-3 text-indigo-400"></i>${file.name}</div>`;
            }
            lucide.createIcons();
        }

        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('file-input');
            if (fileInput.files.length === 0) return alert("Select at least one document to index.");

            const container = document.getElementById('upload-status-container');
            const formData = new FormData();
            
            for (let file of fileInput.files) {
                formData.append("files", file);
                container.insertAdjacentHTML('beforeend', `
                    <div id="toast-${file.name.replace(/[^a-zA-Z0-9]/g, '')}" class="text-[11px] bg-slate-900 border border-slate-800 p-2 rounded-xl flex items-center justify-between">
                        <span class="truncate font-mono text-slate-300">${file.name}</span>
                        <span class="text-indigo-400 animate-pulse shrink-0 ml-2">Indexing...</span>
                    </div>
                `);
            }

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const data = await response.json();
                
                fileInput.value = '';
                document.getElementById('selected-files-list').innerHTML = '';

                data.results.forEach(res => {
                    const id = "toast-" + res.filename.replace(/[^a-zA-Z0-9]/g, '');
                    const element = document.getElementById(id);
                    if (element) {
                        element.className = "text-[11px] bg-emerald-950/20 border border-emerald-800/40 p-2 rounded-xl flex flex-col gap-0.5";
                        element.innerHTML = `<div class="text-emerald-400 font-medium truncate font-mono">${res.filename} Indexed</div>`;
                    }
                });
                
                fetchDatabaseStats();
            } catch (err) {
                alert("Vector processing pipeline execution exception thrown.");
            }
        });

        document.getElementById('chat-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const inputEl = document.getElementById('chat-input');
            const query = inputEl.value.trim();
            if (!query) return;

            const chatWindow = document.getElementById('chat-window');
            const loadingIndicator = document.getElementById('chat-loading');
            
            chatWindow.insertAdjacentHTML('beforeend', `
                <div class="flex items-start gap-3 justify-end">
                    <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none px-4 py-3 max-w-[85%] text-sm shadow-md">
                        <p class="leading-relaxed">${escapeHtml(query)}</p>
                    </div>
                </div>
            `);
            
            inputEl.value = '';
            chatWindow.scrollTop = chatWindow.scrollHeight;
            loadingIndicator.classList.remove('hidden');

            try {
                const formData = new FormData();
                formData.append('question', query);

                const response = await fetch('/ask', { method: 'POST', body: formData });
                const data = await response.json();

                loadingIndicator.classList.add('hidden');

                chatWindow.insertAdjacentHTML('beforeend', `
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 rounded-full bg-indigo-950 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
                            <i data-lucide="bot" class="w-4 h-4"></i>
                        </div>
                        <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none px-4 py-3 max-w-[85%] text-sm text-slate-200 shadow-sm leading-relaxed">
                            <p>${escapeHtml(data.answer)}</p>
                        </div>
                    </div>
                `);

                document.getElementById('inspector-source').textContent = data.source;
                document.getElementById('inspector-context').value = data.context;
                lucide.createIcons();

            } catch(err) {
                loadingIndicator.classList.add('hidden');
                alert("Runtime chat interaction pipeline error encountered.");
            }
            chatWindow.scrollTop = chatWindow.scrollHeight;
        });

        document.getElementById('clear-db-btn').addEventListener('click', async () => {
            if (!confirm("Are you completely sure you want to drop the current collection index? This action cannot be undone.")) return;
            
            try {
                const res = await fetch('/clear-db', { method: 'POST' });
                const data = await res.json();
                alert(data.status);
                
                document.getElementById('inspector-source').textContent = "None";
                document.getElementById('inspector-context').value = "";
                document.getElementById('upload-status-container').innerHTML = "";
                
                document.getElementById('chat-window').innerHTML = `
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 rounded-full bg-indigo-950 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
                        <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none px-4 py-3 max-w-[85%] text-sm text-slate-300">Database collection reset. Ask a question or index new assets.</div>
                    </div>
                `;
                lucide.createIcons();
                fetchDatabaseStats();
            } catch (err) {
                alert("Failed to securely drop the database vector instance.");
            }
        });

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard_ui(request: Request):
    return HTML_UI

@app.get("/stats")
def get_database_stats():
    try:
        collection_data = collection.get(include=["metadatas"])
        total_chunks = len(collection_data["ids"])
        
        unique_docs = set()
        if collection_data["metadatas"]:
            for meta in collection_data["metadatas"]:
                if meta and "source" in meta:
                    unique_docs.add(meta["source"])
                    
        return {
            "documents_count": len(unique_docs),
            "chunks_count": total_chunks,
            "documents_list": list(unique_docs),
            "embedding_model": "Chroma Default (all-MiniLM-L6-v2)"
        }
    except Exception:
        logger.exception("Error extracting live vector collection metrics.")
        return {"documents_count": 0, "chunks_count": 0, "documents_list": [], "embedding_model": "Error reading index parameters."}

@app.post("/upload")
def process_document_uploads(files: List[UploadFile] = File(...)):
    upload_responses = []
    for file in files:
        try:
            result_msg = add_uploaded_file_to_index(file)
            upload_responses.append({"filename": file.filename, "status": result_msg})
        except Exception:
            logger.exception(f"Critical error during file upload: {file.filename}")
            upload_responses.append({"filename": file.filename, "status": "Internal pipeline error."})
            
    return {"results": upload_responses}

@app.post("/ask")
def process_rag_pipeline(question: str = Form(...)):
    if not question.strip():
        return JSONResponse(status_code=400, content={"error": "Question parameter blank."})
        
    try:
        context, sources = retrieve_context(collection, question)
        answer = ask_llm(question, context)
        
        return {
            "answer": answer,
            "context": context if context else "No context blocks matched vector similarity thresholds.",
            "source": sources if sources else "None"
        }
    except Exception:
        logger.exception("RAG pipeline processing loop aborted.")
        return JSONResponse(status_code=500, content={"error": "Internal assistant execution failure."})

@app.post("/clear-db")
def clear_database_index():
    global collection
    try:
        client = get_chroma_client()
        try:
            client.delete_collection("documents")
        except Exception:
            pass
        
        collection = get_collection()
        return {"status": "Database successfully wiped and re-initialized."}
    except Exception:
        logger.exception("Error encountered while attempting to drop vector storage space.")
        return JSONResponse(status_code=500, content={"error": "Failed to reset database safely."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)