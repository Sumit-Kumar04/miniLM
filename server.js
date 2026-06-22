const express = require('express');
const cors = require('cors');
const path = require('path');
const VectorDBEngine = require('./javaBridge');

const app = express();
app.use(cors());
app.use(express.json());

// Routes mapping 1-to-1 to the Java httpServer endpoints

// GET /items -> return list of items
app.get('/items', (req, res) => {
    try {
        res.json(VectorDBEngine.listItems());
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// GET /search -> search for closest vectors
app.get('/search', (req, res) => {
    try {
        const { v, k, metric, algo } = req.query;
        const limit = parseInt(k) || 5;
        const m = metric || 'cosine';
        const a = algo || 'hnsw';
        res.json(VectorDBEngine.search(v || '', limit, m, a));
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// POST /insert -> insert a vector item
app.post('/insert', (req, res) => {
    try {
        const { metadata, category, embedding } = req.body;
        const embStr = Array.isArray(embedding) ? embedding.join(',') : '';
        const id = VectorDBEngine.insert(metadata || '', category || '', embStr);
        res.json({ id });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// DELETE /delete/:id -> delete a vector item
app.delete('/delete/:id', (req, res) => {
    try {
        const id = parseInt(req.params.id);
        const ok = VectorDBEngine.deleteItem(id);
        res.json({ ok });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// GET /benchmark -> benchmark search algorithms
app.get('/benchmark', (req, res) => {
    try {
        const { v, k, metric } = req.query;
        const limit = parseInt(k) || 5;
        const m = metric || 'cosine';
        res.json(VectorDBEngine.benchmark(v || '', limit, m));
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// GET /hnsw-info -> return HNSW layers info
app.get('/hnsw-info', (req, res) => {
    try {
        res.json(VectorDBEngine.hnswInfo());
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// POST /doc/insert -> insert document text (chunks and embeds via Ollama)
app.post('/doc/insert', (req, res) => {
    try {
        const { title, text } = req.body;
        res.json(VectorDBEngine.docInsert(title || '', text || ''));
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// DELETE /doc/delete/:id -> delete document/chunk
app.delete('/doc/delete/:id', (req, res) => {
    try {
        const id = parseInt(req.params.id);
        const ok = VectorDBEngine.docDelete(id);
        res.json({ ok });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// GET /doc/list -> list all stored documents/chunks
app.get('/doc/list', (req, res) => {
    try {
        res.json(VectorDBEngine.docList());
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// POST /doc/search -> search for relevant context using question embedding
app.post('/doc/search', (req, res) => {
    try {
        const { question, k } = req.body;
        const limit = parseInt(k) || 3;
        res.json(VectorDBEngine.docSearch(question || '', limit));
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// POST /doc/ask -> RAG asking using question embedding and Ollama generation
app.post('/doc/ask', (req, res) => {
    try {
        const { question, k } = req.body;
        const limit = parseInt(k) || 3;
        res.json(VectorDBEngine.docAsk(question || '', limit));
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// GET /status -> status of backend (ollama, counts, dimensions)
app.get('/status', (req, res) => {
    try {
        res.json(VectorDBEngine.status());
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// GET /stats -> stats of frontend categories
app.get('/stats', (req, res) => {
    try {
        res.json(VectorDBEngine.stats());
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

// Serve index.html statically for root path
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

const PORT = 8090;

if (require.main === module) {
    app.listen(PORT, () => {
        console.log(`Backend server running at http://localhost:${PORT}`);
    });
}

module.exports = app;
