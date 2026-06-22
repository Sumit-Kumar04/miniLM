const fs = require('fs');
const path = require('path');

function detectJavaHome() {
  if (process.env.JAVA_HOME) return;

  const candidates = [
    'C:\\Program Files\\Java\\jdk-23',
    'C:\\Program Files\\Java\\jdk-21',
    'C:\\Program Files\\Java\\jdk-17',
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      process.env.JAVA_HOME = candidate;
      return;
    }
  }
}

detectJavaHome();

const java = require('java-bridge');
java.classpath.append(path.resolve(__dirname));

const VectorDBEngine = java.importClass('VectorDBEngine');
VectorDBEngine.initSync();

function parseJsonResult(value) {
  if (value == null) return value;
  const text = typeof value === 'string'
    ? value
    : (typeof value.toStringSync === 'function' ? value.toStringSync() : String(value));
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

module.exports = {
  listItems: () => parseJsonResult(VectorDBEngine.listItemsSync()),
  search: (vecStr, k, metric, algo) => parseJsonResult(VectorDBEngine.searchSync(vecStr, k, metric, algo)),
  insert: (meta, cat, embStr) => VectorDBEngine.insertSync(meta, cat, embStr),
  deleteItem: (id) => VectorDBEngine.deleteItemSync(id),
  benchmark: (vecStr, k, metric) => parseJsonResult(VectorDBEngine.benchmarkSync(vecStr, k, metric)),
  hnswInfo: () => parseJsonResult(VectorDBEngine.hnswInfoSync()),
  docInsert: (title, text) => parseJsonResult(VectorDBEngine.docInsertSync(title, text)),
  docDelete: (id) => VectorDBEngine.docDeleteSync(id),
  docList: () => parseJsonResult(VectorDBEngine.docListSync()),
  docSearch: (question, k) => parseJsonResult(VectorDBEngine.docSearchSync(question, k)),
  docAsk: (question, k) => parseJsonResult(VectorDBEngine.docAskSync(question, k)),
  status: () => parseJsonResult(VectorDBEngine.statusSync()),
  stats: () => parseJsonResult(VectorDBEngine.statsSync()),
};
