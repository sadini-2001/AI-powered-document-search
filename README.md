# 🧠 AI-Powered Document Search

A lightweight **Flask + FAISS + HuggingFace** web app that lets you upload documents (PDF/TXT) and perform **semantic search** over their content.  

🚀 Built to demonstrate skills in **NLP, embeddings, vector databases, and full-stack ML integration**.  

---

## ✨ Features
- 📤 Upload **PDFs or TXT files**  
- 🔍 Perform **semantic search** using `all-mpnet-base-v2` embeddings  
- 📊 View **top 10 keywords** across uploaded docs (analytics tab)  
- 🗑️ Delete uploaded files → auto rebuilds FAISS index  
- 🎨 **Bootstrap UI** for clean dark-themed interface  

---

## ⚡ Installation

Clone the repo:
```bash
git clone https://github.com/<your-username>/AI-Document-Search.git
cd AI-Document-Search
```

---

## Create a virtual environment:
```bash
python -m venv .venv
# On Linux/Mac
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

---

## Install dependencies:
```bash
pip install -r requirements.txt


---

## ▶️ Run the App
```bash
python app.py

---

## Project Structure
AI-Document-Search/
│── app.py                 # Main Flask app
│── requirements.txt       # Dependencies
│── README.md              # Project overview
│── .gitignore             # Ignore runtime/IDE files
│
├── templates/
   └── index.html         # Bootstrap + Jinja2 template

---

## 🛠️ Tech Stack

- Flask (backend framework)
- LangChain + FAISS (vector search engine)
- HuggingFace Sentence Transformers (all-mpnet-base-v2 for embeddings)
- SQLite (store metadata about uploaded docs)
- Bootstrap 5 (dark-themed frontend)

---

## 🎯 Example Use Cases

Search policy documents for exact clauses
Explore large research papers
Build knowledge-base Q&A systems
Lightweight demo for semantic search pipelines

---

# 🧠 AI-Powered Document Search

A lightweight **Flask + FAISS + HuggingFace** web app that lets you upload documents (PDF/TXT) and perform **semantic search** over their content.  

🚀 Built to demonstrate skills in **NLP, embeddings, vector databases, and full-stack ML integration**.  

---

## ✨ Features
- 📤 Upload **PDFs or TXT files**  
- 🔍 Perform **semantic search** using `all-mpnet-base-v2` embeddings  
- 📊 View **top 10 keywords** across uploaded docs (analytics tab)  
- 🗑️ Delete uploaded files → auto rebuilds FAISS index  
- 🎨 **Bootstrap UI** for clean dark-themed interface  

---

## ⚡ Installation

Clone the repo:
```bash
git clone https://github.com/<your-username>/AI-Document-Search.git
cd AI-Document-Search

---

## Create a virtual environment:
```bash
python -m venv .venv
# On Linux/Mac
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

---

## Install dependencies:
```bash
pip install -r requirements.txt


---

## ▶️ Run the App
```bash
python app.py

---

## Project Structure
AI-Document-Search/
│── app.py                 # Main Flask app
│── requirements.txt       # Dependencies
│── README.md              # Project overview
│── .gitignore             # Ignore runtime/IDE files
│
├── templates/
   └── index.html         # Bootstrap + Jinja2 template

---

## 🛠️ Tech Stack

- Flask (backend framework)
- LangChain + FAISS (vector search engine)
- HuggingFace Sentence Transformers (all-mpnet-base-v2 for embeddings)
- SQLite (store metadata about uploaded docs)
- Bootstrap 5 (dark-themed frontend)

---

## 🎯 Example Use Cases

Search policy documents for exact clauses
Explore large research papers
Build knowledge-base Q&A systems
Lightweight demo for semantic search pipelines

---

🚀 Future Improvements

Integrate MMR retrieval for more diverse results

Add cross-encoder reranking for higher accuracy

Deploy on Docker/Cloud for production use






