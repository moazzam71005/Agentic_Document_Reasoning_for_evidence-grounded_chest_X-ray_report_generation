# Layout-Aware RAG

This note covers the hierarchical PDF parsing and retrieval approach.

## What
A hybrid retrieval system (BM25 + FAISS) querying guideline PDFs that have been parsed into a hierarchical structure (`{id, parent_id, title_path, sibling_idx}`). Retrieved passages are expanded to include structural sibling paragraphs.

## Why
Clinical guidelines (e.g. Fleischner glossary) contain definitions and tables where context is heavily dependent on headers and structural layout. Standard flat chunking (e.g. splitting by character limit) splits tables and breaks parent-child context, rendering retrieved snippets incomprehensible or misleading.

## Alternatives Considered & Outcome
* **Flat Sentence/Token Chunking**: Simple token-based splitting. Tested and discarded due to context fragmentation and failure to capture parent-scope headers.
* **Outcome**: Implemented layout-aware parsing using structural hierarchy with BM25 + dense fusion (using `all-MiniLM-L6-v2` embeddings) and parent/sibling context expansion.

---
*Related: [[decisions-moc]]*
