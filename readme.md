# Applied AI Engineering Workshop

A hands-on Python workshop that builds AI applications in small, testable steps. Learners move from text embeddings and semantic search to retrieval-augmented generation (RAG), LangChain pipelines, conversation memory, tools, and agent workflows.

The mentor guide's teaching loop is: frame a problem, predict what should happen, implement one small change, inspect and explain it, run a test, challenge it with an edge case, and reflect on the data flow.

## Workshop learning path

| Session | Topics |
| --- | --- |
| 1. Embeddings and RAG | Text embeddings, vector databases, semantic search, RAG, and a knowledge-base chatbot assignment |
| 2. Pipelines and tools | LangChain Expression Language (LCEL), conversation memory, custom tools, and an assistant-tool assignment |
| 3. Agents and workflows | Tool-calling agents, stateful LangGraph workflows, and a domain-specific agent assignment |

## Repository contents

```text
.
├── data/
│   └── AI_Ethics_Policy.pdf
├── labs/
│   ├── demo_1_vector_db.py
│   └── demo_2_rag_chatbot.py
├── requirements.txt
└── readme.md
```

The workspace currently includes the first two runnable labs:

- **`labs/demo_1_vector_db.py`** — creates embeddings with `all-MiniLM-L6-v2`, compares sentence similarity, stores sample documents in ChromaDB, and retrieves documents for a semantic query.
- **`labs/demo_2_rag_chatbot.py`** — extracts text from the sample PDF, chunks and embeds it, retrieves relevant passages, and asks Gemini to answer using that evidence.

The mentor guide also covers LCEL, memory, tools, LangGraph, and email-agent examples. Those later demos are not currently included as scripts in `labs/`.

## Requirements

- Python 3.12 is the interpreter used for this workspace.
- A Google Gemini API key is required for the RAG chatbot. The vector database demo uses a local embedding model; its first run downloads that model.
- Internet access is needed to install packages, download the embedding model, and call Gemini.

## Setup

Run these commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, open a new PowerShell session and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If creating a virtual environment on Windows fails with `spawn EPERM`, select an installed CPython interpreter rather than the Microsoft Store alias under `WindowsApps`, then retry environment creation. In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**.

## Configure the Gemini API key

Create or update a `.env` file in the repository root:

```dotenv
GOOGLE_API_KEY=your_gemini_api_key
```

Keep the key private. `.env` is ignored by Git; do not paste API keys into source files, commits, or AI chat prompts.

## Run the labs

Run from the repository root so the relative PDF and database paths resolve correctly.

```powershell
python labs\demo_1_vector_db.py
python labs\demo_2_rag_chatbot.py
```

The first lab stores its local ChromaDB collection in `chroma_db/`. The second uses `data/AI_Ethics_Policy.pdf` and stores its collection in `rag_chroma/`. These are generated local data directories, not source files.

## Teaching and safety notes

- Build incrementally; do not ask an AI assistant to generate the entire application in one step.
- Review generated imports, inputs, outputs, validation, and dependencies before running code.
- Inspect retrieved passages and test whether each RAG answer is actually supported by its evidence. RAG can reduce unsupported answers, but does not guarantee correctness.
- Keep secrets in environment variables and use a dedicated test account for any future email integration.
- For email-agent exercises, prefer mock data. Keep real integrations read-only or draft-only, require human review, and do not expose send, delete, archive, or modify actions.

## Troubleshooting

- **Gemini authentication or access error:** Check that `GOOGLE_API_KEY` is set in the root `.env` file and that the configured model is available to your account.
- **PDF text is empty:** `pypdf` extracts selectable text but does not perform OCR; use a text-based PDF or OCR the document first.
- **RAG retrieves irrelevant passages:** Inspect the extracted text and chunks, then experiment with chunk size and the number of retrieved passages (`k`).
- **Chroma embedding dimension error:** Use the same embedding model when storing and querying a collection. Rebuild the local demo collection if you change models.
- **Package/API differences:** LangChain APIs evolve. Check installed package versions and current documentation if an import or helper differs from the workshop notes.
