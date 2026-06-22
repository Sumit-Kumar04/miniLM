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

<img width="1901" height="1078" alt="Screenshot 2026-06-22 223833" src="https://github.com/user-attachments/assets/4e5bf718-96a5-4524-92cf-7d678703af77" />
<img width="1918" height="1078" alt="Screenshot 2026-06-22 223756" src="https://github.com/user-attachments/assets/9bf59e5e-d118-4491-a4a2-7d3442d2ce7d" />
<img width="1917" height="1078" alt="Screenshot 2026-06-22 223809" src="https://github.com/user-attachments/assets/37b1e2f4-1293-42d4-bded-3486429d0781" />


