# 📖 The Archive — Chat with Your Book (RAG App)

A Retrieval-Augmented Generation (RAG) app that lets you upload any PDF book or
document and have a conversation with it. Ask questions in plain English and
get answers grounded strictly in the content of the document — no
hallucinated facts, no answers pulled from outside the text.

Built with **LangChain**, **Mistral AI**, **ChromaDB**, and a custom
**Streamlit** interface.

> 🎓 Built while following a RAG tutorial from **Sheryians Coding School**,
> then extended with a custom-styled Streamlit UI and PDF upload workflow.

---

## ✨ Features

- **Upload any PDF** directly from the browser — no manual scripts to run
- **Automatic chunking + embedding** of the document using `mistral-embed`
- **Persistent vector storage** with ChromaDB (the database survives app restarts)
- **MMR-based retrieval** (Maximal Marginal Relevance) for more diverse,
  less redundant context instead of plain top-k similarity search
- **Context-grounded answers** — the LLM is instructed to answer only from
  retrieved chunks, and to say so plainly when the answer isn't in the document
- **Chat-style interface** with conversation history, built with Streamlit's
  native chat components
- Custom dark "night library" themed UI

---

## 🧱 Tech Stack

| Layer            | Tool                                    |
|-------------------|------------------------------------------|
| LLM                | Mistral (`mistral-small-2603`) via `langchain-mistralai` |
| Embeddings         | Mistral (`mistral-embed`)                |
| Vector Store       | ChromaDB (local, persisted to disk)      |
| Orchestration      | LangChain                                |
| PDF Parsing        | `langchain_community.document_loaders.PyPDFLoader` |
| UI                 | Streamlit                                |
| Env management     | `python-dotenv`                          |

---

## 🗂️ Project Structure

```
.
├── app.py                # Streamlit UI — upload a PDF, chat with it
├── main.py                # CLI version — chat with an already-built database
├── create_database.py     # Script version — build the vector DB from a fixed PDF path
├── chroma_db/              # Persisted vector database (created automatically)
├── .env                    # Your API keys (not committed to version control)
└── README.md
```

- **`app.py`** is the recommended way to use this project — it combines the
  database-building step and the chat step into one interactive UI.
- **`create_database.py`** and **`main.py`** are the original standalone
  scripts this project was built from (build the DB once via the terminal,
  then chat via the terminal).

---

## ⚙️ How It Works

1. **Load** — the PDF is parsed page-by-page using `PyPDFLoader`.
2. **Split** — the text is broken into overlapping chunks
   (`chunk_size=1000`, `chunk_overlap=20`) using `RecursiveCharacterTextSplitter`,
   so context isn't lost at chunk boundaries.
3. **Embed** — each chunk is converted into a vector using Mistral's
   `mistral-embed` model.
4. **Store** — the vectors are saved to a local ChromaDB collection on disk
   (`chroma_db/`), so the database doesn't need to be rebuilt every session.
5. **Retrieve** — when you ask a question, the retriever uses **MMR search**
   (`k=4`, `fetch_k=10`, `lambda_mult=0.5`) to pull the most relevant *and*
   diverse chunks from the database.
6. **Answer** — the retrieved chunks are inserted into a strict prompt
   template and sent to the LLM, which is instructed to answer *only* from
   that context — or say it couldn't find the answer.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS/Linux

pip install streamlit langchain langchain-community langchain-mistralai chromadb pypdf python-dotenv
```

### 3. Add your API key

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit gives you, upload a PDF from the sidebar,
click **Process & Create Database**, and start asking questions.

---

## 🖥️ Alternative: Command-Line Usage

If you'd rather not use the UI:

```bash
# 1. Build the database from a PDF (edit the path inside the script first)
python create_database.py

# 2. Chat with it in the terminal
python main.py
```

---

## 📌 Notes & Limitations

- Uploading a new PDF through the UI **overwrites** the existing `chroma_db`
  (one active book at a time, by design).
- Answers are intentionally restricted to the uploaded document's content —
  this is a feature, not a bug, for factual/document Q&A use cases.
- Large PDFs will take longer to process on first upload since every chunk
  needs to be embedded.

---

## 🙏 Acknowledgements

This project was built while following a RAG tutorial by
[Sheryians Coding School](https://www.sheryians.com/), with the Streamlit UI,
styling, and upload workflow added on top as an extension of the original
tutorial code.

---

## 📄 License

Feel free to use, modify, and learn from this project.
