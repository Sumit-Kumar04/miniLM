# MiniLM Vector DB

Hybrid Java + Node.js vector database demo with a browser UI.

## What it does

- Stores and searches vectors in Java
- Exposes the Java engine through a Node.js bridge
- Provides a browser interface for search, insert, delete, benchmark, and document RAG
- Uses Ollama for embeddings and answer generation in document features

## Tech Stack

- Java
- Node.js
- Express
- `java-bridge`
- HTML, CSS, JavaScript
- Ollama

## Files

- `VectorDBEngine.java` - core vector and document logic
- `javaBridge.js` - Java bridge wrapper for Node.js
- `server.js` - Express API server
- `index.html` - frontend UI
- `docs.md` - detailed internal documentation and interview prep

## Requirements

- Node.js installed
- Java JDK installed
- Ollama running locally for document/RAG features

If `JAVA_HOME` is not set, the app tries common Windows JDK locations automatically.

## Run

```powershell
npm install
npm start
```

Open:

```text
http://localhost:8090
```

## Java Notes

If you change `VectorDBEngine.java`, recompile it so the bridge can load the updated class.

## API Overview

- `GET /items`
- `GET /search`
- `POST /insert`
- `DELETE /delete/:id`
- `GET /benchmark`
- `GET /hnsw-info`
- `POST /doc/insert`
- `DELETE /doc/delete/:id`
- `GET /doc/list`
- `POST /doc/search`
- `POST /doc/ask`
- `GET /status`
- `GET /stats`

## Notes

- The project is currently in-memory; restarting clears stored vectors and documents.
- `docs.md` is intentionally ignored by git and is meant as local project documentation.

