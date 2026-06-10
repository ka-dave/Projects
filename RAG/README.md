# RAG

End-to-end Retrieval-Augmented Generation — from learning the fundamentals to a production-grade system.

## Structure

```
RAG/
├── TRAG/          # Learning projects: RAG concepts explored stage by stage
└── (coming soon)  # Production RAG system
```

### Learning (`TRAG/`)

Isolated experiments covering each stage of a RAG pipeline: document ingestion, chunking, embeddings, retrieval, and generation. Each project focuses on one concept so it can be understood before being composed into the full system.

See [TRAG/README.md](./TRAG/README.md) for details.

### Production (planned)

A full end-to-end RAG system built on the concepts from the learning phase. Will cover:

- Ingestion pipeline with chunking and embedding
- Vector store indexing and hybrid retrieval
- LLM-powered answer generation with source grounding
- Evaluation (retrieval quality, answer faithfulness)
- Production concerns: caching, re-ranking, observability, latency

## RAG Pipeline Stages

```
Documents → Ingestion → Chunking → Embedding → Vector Store
                                                     ↓
                                   Query → Retrieval → LLM → Answer
```
