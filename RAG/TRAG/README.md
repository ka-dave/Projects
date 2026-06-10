# TRAG — Document Ingestion

Learning project covering the first stage of a RAG pipeline: getting documents into a format the system can work with.

## What it covers

- Creating `Document` objects with metadata (source, author, date, page count)
- Loading `.txt` files from a directory using `DirectoryLoader` + `TextLoader`
- Loading PDF files using `PyMuPDFLoader`
- Exploring document structure and content in Jupyter notebooks

## Setup

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter notebook notebook/document.ipynb
```

The notebook walks through ingesting the sample data in `data/` — a couple of `.txt` files and a PDF deck on RAG concepts.

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core RAG abstractions |
| `langchain-community` | Document loaders (`DirectoryLoader`, `PyMuPDFLoader`) |
| `langchain-core` | `Document` type and base interfaces |
| `pypdf` | PDF parsing |
| `pymupdf` | Fast PDF parsing (used by `PyMuPDFLoader`) |
| `ipykernel` | Jupyter notebook support |

## Data

Sample files used during exploration live in `data/`:

```
data/
├── text_files/
│   ├── python_intro.txt
│   └── machine_learning.txt
└── pdf/
    └── rag-deck.pdf
```

> [!NOTE]
> This is exploratory learning code. It reflects understanding at this point in the RAG learning journey.
