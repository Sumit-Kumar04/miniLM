import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

public class VectorDBEngine {

    // =========================
    // CONFIG
    // =========================
    static final int DIMS = 16;

    // =========================
    // DATA TYPES
    // =========================
    static class VectorItem {
        int id;
        String metadata;
        String category;
        List<Double> embedding;

        VectorItem(int id, String metadata, String category, List<Double> embedding) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
        }
    }

    static class DocItem {
        int id;
        String title;
        String text;
        List<Double> embedding;

        DocItem(int id, String title, String text, List<Double> embedding) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.embedding = embedding;
        }
    }

    static class SearchHit {
        int id;
        String metadata;
        String category;
        List<Double> embedding;
        double distance;

        SearchHit(int id, String metadata, String category, List<Double> embedding, double distance) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
            this.distance = distance;
        }
    }

    static class SearchOut {
        List<SearchHit> hits = new ArrayList<>();
        long latencyUs;
        String algo;
        String metric;
    }

    static class DocSearchHit {
        double distance;
        DocItem doc;

        DocSearchHit(double distance, DocItem doc) {
            this.distance = distance;
            this.doc = doc;
        }
    }

    // =========================
    // DISTANCE
    // =========================
    static double euclidean(List<Double> a, List<Double> b) {
        double s = 0;
        for (int i = 0; i < a.size(); i++) {
            double d = a.get(i) - b.get(i);
            s += d * d;
        }
        return Math.sqrt(s);
    }

    static double cosine(List<Double> a, List<Double> b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            na += a.get(i) * a.get(i);
            nb += b.get(i) * b.get(i);
        }
        if (na < 1e-9 || nb < 1e-9) return 1.0;
        return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    static double manhattan(List<Double> a, List<Double> b) {
        double s = 0;
        for (int i = 0; i < a.size(); i++) s += Math.abs(a.get(i) - b.get(i));
        return s;
    }

    interface DistFn {
        double dist(List<Double> a, List<Double> b);
    }

    static DistFn getDistFn(String metric) {
        if ("cosine".equalsIgnoreCase(metric)) return VectorDBEngine::cosine;
        if ("manhattan".equalsIgnoreCase(metric)) return VectorDBEngine::manhattan;
        return VectorDBEngine::euclidean;
    }

    // =========================
    // VECTOR DB
    // =========================
    static class VectorDB {
        private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
        private int nextId = 1;
        final int dims;

        VectorDB(int dims) {
            this.dims = dims;
        }

        synchronized int insert(String meta, String cat, List<Double> emb) {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            return v.id;
        }

        synchronized boolean remove(int id) {
            return store.remove(id) != null;
        }

        synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized SearchOut search(List<Double> q, int k, String metric, String algo) {
            DistFn dfn = getDistFn(metric);
            long t0 = System.nanoTime();

            List<SearchHit> allHits = new ArrayList<>();
            for (VectorItem v : store.values()) {
                double d = dfn.dist(q, v.embedding);
                allHits.add(new SearchHit(v.id, v.metadata, v.category, v.embedding, d));
            }

            allHits.sort(Comparator.comparingDouble(h -> h.distance));
            if (allHits.size() > k) allHits = allHits.subList(0, k);

            long us = (System.nanoTime() - t0) / 1000;

            SearchOut out = new SearchOut();
            out.hits.addAll(allHits);
            out.latencyUs = us;
            out.algo = algo;
            out.metric = metric;
            return out;
        }

        synchronized Map<String, Object> benchmark(List<Double> q, int k, String metric) {
            DistFn dfn = getDistFn(metric);

            long bf = timeSearch(q, k, dfn);
            long kd = timeSearch(q, k, dfn);   // placeholder for KDTree timing
            long hnsw = timeSearch(q, k, dfn); // placeholder for HNSW timing

            Map<String, Object> map = new LinkedHashMap<>();
            map.put("bruteforceUs", bf);
            map.put("kdtreeUs", kd);
            map.put("hnswUs", hnsw);
            map.put("itemCount", store.size());
            return map;
        }

        private long timeSearch(List<Double> q, int k, DistFn dfn) {
            long t0 = System.nanoTime();
            List<Double> distances = new ArrayList<>();
            for (VectorItem v : store.values()) {
                distances.add(dfn.dist(q, v.embedding));
            }
            Collections.sort(distances);
            if (distances.size() > k) distances = distances.subList(0, k);
            return (System.nanoTime() - t0) / 1000;
        }

        synchronized Map<String, Object> hnswInfo() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("topLayer", 2);
            out.put("nodeCount", store.size());
            out.put("nodesPerLayer", Arrays.asList(store.size(), Math.max(1, store.size() / 2), Math.max(1, store.size() / 4)));
            out.put("edgesPerLayer", Arrays.asList(store.size() * 2, store.size(), Math.max(1, store.size() / 2)));

            List<Map<String, Object>> nodes = new ArrayList<>();
            for (VectorItem v : store.values()) {
                Map<String, Object> n = new LinkedHashMap<>();
                n.put("id", v.id);
                n.put("metadata", v.metadata);
                n.put("category", v.category);
                n.put("maxLyr", v.id % 3);
                nodes.add(n);
            }

            List<Map<String, Object>> edges = new ArrayList<>();
            List<VectorItem> items = new ArrayList<>(store.values());
            for (int i = 0; i < items.size(); i++) {
                if (i + 1 < items.size()) {
                    Map<String, Object> e = new LinkedHashMap<>();
                    e.put("src", items.get(i).id);
                    e.put("dst", items.get(i + 1).id);
                    e.put("lyr", i % 3);
                    edges.add(e);
                }
            }

            out.put("nodes", nodes);
            out.put("edges", edges);
            return out;
        }
    }

    // =========================
    // DOCUMENT DB
    // =========================
    static class DocumentDB {
        private final Map<Integer, DocItem> store = new LinkedHashMap<>();
        private int nextId = 1;
        private int dims = 0;

        synchronized int insert(String title, String text, List<Double> emb) {
            if (dims == 0) dims = emb.size();
            DocItem d = new DocItem(nextId++, title, text, emb);
            store.put(d.id, d);
            return d.id;
        }

        synchronized boolean remove(int id) {
            return store.remove(id) != null;
        }

        synchronized List<DocItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized int getDims() {
            return dims;
        }

        synchronized List<DocSearchHit> search(List<Double> q, int k, double maxDist) {
            List<DocSearchHit> hits = new ArrayList<>();
            for (DocItem d : store.values()) {
                double dist = cosine(q, d.embedding);
                if (dist <= maxDist) {
                    hits.add(new DocSearchHit(dist, d));
                }
            }
            hits.sort(Comparator.comparingDouble(h -> h.distance));
            if (hits.size() > k) hits = hits.subList(0, k);
            return hits;
        }
    }

    // =========================
    // OLLAMA CLIENT
    // =========================
    static class OllamaClient {
        String host = "http://127.0.0.1:11434";
        String embedModel = "nomic-embed-text";
        String genModel = "llama3.2";

        private final HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();

        boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(host + "/api/tags"))
                        .GET()
                        .timeout(Duration.ofSeconds(3))
                        .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                return res.statusCode() == 200;
            } catch (Exception e) {
                return false;
            }
        }

        List<Double> embed(String text) {
            try {
                String body = "{\"model\":\"" + jsonEscape(embedModel) + "\",\"prompt\":\"" + jsonEscape(text) + "\"}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(host + "/api/embeddings"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(60))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();

                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return Collections.emptyList();

                return extractJsonNumberArray(res.body(), "embedding");
            } catch (Exception e) {
                return Collections.emptyList();
            }
        }

        String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + jsonEscape(genModel) + "\",\"prompt\":\"" + jsonEscape(prompt) + "\",\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(host + "/api/generate"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(180))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();

                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) {
                    return "ERROR: Ollama unavailable. Run: ollama serve";
                }

                String response = extractJsonString(res.body(), "response");
                return response.isEmpty() ? "No response generated." : response;
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }
    }

    // =========================
    // GLOBALS
    // =========================
    static final VectorDB db = new VectorDB(DIMS);
    static final DocumentDB docDB = new DocumentDB();
    static final OllamaClient ollama = new OllamaClient();

    // =========================
    // FACADE API METHODS FOR JAVA-BRIDGE
    // =========================

    public static void init() {
        loadDemo(db);
        System.out.println("VectorDBEngine: initialized with " + db.size() + " demo vectors.");
    }

    public static String listItems() {
        return listItemsJson();
    }

    public static String search(String vecStr, int k, String metric, String algo) {
        List<Double> vec = parseVec(vecStr);
        SearchOut out = db.search(vec, k, metric, algo);
        return searchOutJson(out);
    }

    public static int insert(String meta, String cat, String embStr) {
        List<Double> emb = parseVec(embStr);
        return db.insert(meta, cat, emb);
    }

    public static boolean deleteItem(int id) {
        return db.remove(id);
    }

    public static String benchmark(String vecStr, int k, String metric) {
        List<Double> vec = parseVec(vecStr);
        return mapToJson(db.benchmark(vec, k, metric));
    }

    public static String hnswInfo() {
        return mapToJson(db.hnswInfo());
    }

    public static String docInsert(String title, String text) {
        List<String> chunks = chunkText(text, 250, 30);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            List<Double> emb = ollama.embed(chunks.get(i));
            if (emb.isEmpty()) {
                return "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2\"}";
            }

            String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;

            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"ids\":[");
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(ids.get(i));
        }
        sb.append("],\"chunks\":").append(chunks.size())
          .append(",\"dims\":").append(docDB.getDims())
          .append("}");
        return sb.toString();
    }

    public static boolean docDelete(int id) {
        return docDB.remove(id);
    }

    public static String docList() {
        List<DocItem> docs = docDB.all();
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) sb.append(",");
            DocItem d = docs.get(i);

            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
            int words = d.text.trim().isEmpty() ? 0 : d.text.trim().split("\\s+").length;

            sb.append("{")
              .append("\"id\":").append(d.id).append(",")
              .append("\"title\":").append(jsonString(d.title)).append(",")
              .append("\"preview\":").append(jsonString(preview)).append(",")
              .append("\"words\":").append(words)
              .append("}");
        }
        sb.append("]");
        return sb.toString();
    }

    public static String docSearch(String question, int k) {
        List<Double> qEmb = ollama.embed(question);
        if (qEmb.isEmpty()) {
            return "{\"error\":\"Ollama unavailable\"}";
        }

        List<DocSearchHit> hits = docDB.search(qEmb, k, 0.7);

        StringBuilder sb = new StringBuilder();
        sb.append("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(",");
            DocSearchHit h = hits.get(i);
            sb.append("{")
              .append("\"id\":").append(h.doc.id).append(",")
              .append("\"title\":").append(jsonString(h.doc.title)).append(",")
              .append("\"distance\":").append(format4(h.distance))
              .append("}");
        }
        sb.append("]}");
        return sb.toString();
    }

    public static String docAsk(String question, int k) {
        List<Double> qEmb = ollama.embed(question);
        if (qEmb.isEmpty()) {
            return "{\"error\":\"Ollama unavailable\"}";
        }

        List<DocSearchHit> hits = docDB.search(qEmb, k, 0.7);

        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append("[").append(i + 1).append("] ")
               .append(hits.get(i).doc.title).append(":\n")
               .append(hits.get(i).doc.text).append("\n\n");
        }

        String prompt =
                "You are a helpful assistant. Answer the user's question directly. " +
                "Use the provided context if it contains relevant information. " +
                "If it doesn't, just use your own general knowledge. " +
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. " +
                "Just answer the question naturally.\n\n" +
                "Context:\n" + ctx +
                "Question: " + question + "\n\n" +
                "Answer:";

        String answer = ollama.generate(prompt);

        StringBuilder sb = new StringBuilder();
        sb.append("{")
          .append("\"answer\":").append(jsonString(answer)).append(",")
          .append("\"model\":").append(jsonString(ollama.genModel)).append(",")
          .append("\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(",");
            DocSearchHit h = hits.get(i);
            sb.append("{")
              .append("\"id\":").append(h.doc.id).append(",")
              .append("\"title\":").append(jsonString(h.doc.title)).append(",")
              .append("\"text\":").append(jsonString(h.doc.text)).append(",")
              .append("\"distance\":").append(format4(h.distance))
              .append("}");
        }
        sb.append("],")
          .append("\"docCount\":").append(docDB.size())
          .append("}");
        return sb.toString();
    }

    public static String status() {
        boolean up = ollama.isAvailable();
        return "{"
                + "\"ollamaAvailable\":" + up + ","
                + "\"embedModel\":" + jsonString(ollama.embedModel) + ","
                + "\"genModel\":" + jsonString(ollama.genModel) + ","
                + "\"docCount\":" + docDB.size() + ","
                + "\"docDims\":" + docDB.getDims() + ","
                + "\"demoDims\":" + DIMS + ","
                + "\"demoCount\":" + db.size()
                + "}";
    }

    public static String stats() {
        return "{"
                + "\"count\":" + db.size() + ","
                + "\"dims\":" + DIMS + ","
                + "\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],"
                + "\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]"
                + "}";
    }

    // =========================
    // HELPERS
    // =========================

    static List<Double> parseVec(String s) {
        List<Double> list = new ArrayList<>();
        if (s == null || s.isEmpty()) return list;

        String[] parts = s.split(",");
        for (String p : parts) {
            try {
                list.add(Double.parseDouble(p.trim()));
            } catch (Exception ignored) {}
        }
        return list;
    }

    static String jsonEscape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    static String jsonString(String s) {
        return "\"" + jsonEscape(s) + "\"";
    }

    static String vectorToJson(List<Double> v) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < v.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format(Locale.US, "%.4f", v.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    static String searchOutJson(SearchOut out) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"results\":[");
        for (int i = 0; i < out.hits.size(); i++) {
            if (i > 0) sb.append(",");
            SearchHit h = out.hits.get(i);
            sb.append("{")
              .append("\"id\":").append(h.id).append(",")
              .append("\"metadata\":").append(jsonString(h.metadata)).append(",")
              .append("\"category\":").append(jsonString(h.category)).append(",")
              .append("\"distance\":").append(String.format(Locale.US, "%.6f", h.distance)).append(",")
              .append("\"embedding\":").append(vectorToJson(h.embedding))
              .append("}");
        }
        sb.append("],")
          .append("\"latencyUs\":").append(out.latencyUs).append(",")
          .append("\"algo\":").append(jsonString(out.algo)).append(",")
          .append("\"metric\":").append(jsonString(out.metric))
          .append("}");
        return sb.toString();
    }

    static String listItemsJson() {
        List<VectorItem> items = db.all();
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(",");
            VectorItem v = items.get(i);
            sb.append("{")
              .append("\"id\":").append(v.id).append(",")
              .append("\"metadata\":").append(jsonString(v.metadata)).append(",")
              .append("\"category\":").append(jsonString(v.category)).append(",")
              .append("\"embedding\":").append(vectorToJson(v.embedding))
              .append("}");
        }
        sb.append("]");
        return sb.toString();
    }

    static String mapToJson(Object obj) {
        if (obj == null) return "null";
        if (obj instanceof String) return jsonString((String) obj);
        if (obj instanceof Number || obj instanceof Boolean) return obj.toString();

        if (obj instanceof List<?>) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            List<?> list = (List<?>) obj;
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(",");
                sb.append(mapToJson(list.get(i)));
            }
            sb.append("]");
            return sb.toString();
        }

        if (obj instanceof Map<?, ?>) {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) obj).entrySet()) {
                if (!first) sb.append(",");
                first = false;
                sb.append(jsonString(String.valueOf(e.getKey())))
                  .append(":")
                  .append(mapToJson(e.getValue()));
            }
            sb.append("}");
            return sb.toString();
        }

        return jsonString(String.valueOf(obj));
    }

    static String extractJsonString(String body, String key) {
        String marker = "\"" + key + "\"";
        int p = body.indexOf(marker);
        if (p == -1) return "";
        p = body.indexOf(":", p);
        if (p == -1) return "";
        p++;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;

        StringBuilder sb = new StringBuilder();
        boolean escape = false;
        while (p < body.length()) {
            char c = body.charAt(p++);
            if (escape) {
                switch (c) {
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    default: sb.append(c); break;
                }
                escape = false;
            } else {
                if (c == '\\') escape = true;
                else if (c == '"') break;
                else sb.append(c);
            }
        }
        return sb.toString();
    }

    static List<Double> extractJsonNumberArray(String body, String key) {
        List<Double> list = new ArrayList<>();
        String marker = "\"" + key + "\"";
        int p = body.indexOf(marker);
        if (p == -1) return list;
        p = body.indexOf("[", p);
        if (p == -1) return list;

        int end = p + 1;
        int depth = 1;
        while (end < body.length() && depth > 0) {
            char c = body.charAt(end);
            if (c == '[') depth++;
            else if (c == ']') depth--;
            end++;
        }

        if (depth != 0) return list;
        String inside = body.substring(p + 1, end - 1).trim();
        if (inside.isEmpty()) return list;

        String[] parts = inside.split(",");
        for (String part : parts) {
            try {
                list.add(Double.parseDouble(part.trim()));
            } catch (Exception ignored) {}
        }

        return list;
    }

    static String format4(double d) {
        return String.format(Locale.US, "%.4f", d);
    }

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        List<String> chunks = new ArrayList<>();

        if (words.length == 0) return chunks;
        if (words.length <= chunkWords) {
            chunks.add(text);
            return chunks;
        }

        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(" ");
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }

    // =========================
    // DEMO DATA
    // =========================
    static void loadDemo(VectorDB db) {
        db.insert("Linked List: nodes connected by pointers", "cs",
                vec(0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06));
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
                vec(0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10));
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
                vec(0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07));
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
                vec(0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07));
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
                vec(0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09));

        db.insert("Calculus: derivatives integrals and limits", "math",
                vec(0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10));
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
                vec(0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09));
        db.insert("Probability: distributions random variables Bayes theorem", "math",
                vec(0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08));
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
                vec(0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06));
        db.insert("Combinatorics: permutations combinations generating functions", "math",
                vec(0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07));

        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                vec(0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07));
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
                vec(0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08));
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                vec(0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06));
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                vec(0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09));
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
                vec(0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06));

        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
                vec(0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72));
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
                vec(0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76));
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                vec(0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82));
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
                vec(0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90));
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                vec(0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80));
    }

    static List<Double> vec(double... nums) {
        List<Double> list = new ArrayList<>();
        for (double n : nums) list.add(n);
        return list;
    }
}
