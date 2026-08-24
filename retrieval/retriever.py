import math
import re
import sys
from collections import Counter

import numpy as np


def tokenize_text(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z]+(?:[-_.][a-z0-9]+)*|\d+(?:\.\d+)*", text)
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seq) == 1:
            tokens.append(seq)
        else:
            tokens.extend(seq[i:i + 2] for i in range(len(seq) - 1))
            if len(seq) >= 3:
                tokens.extend(seq[i:i + 3] for i in range(len(seq) - 2))
    return tokens


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = [tokenize_text(x) for x in documents]
        self.k1 = k1
        self.b = b
        self.n = len(self.documents)
        self.lengths = np.asarray([len(x) for x in self.documents], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.documents]
        df = Counter()
        for tokens in self.documents:
            df.update(set(tokens))
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query):
        if self.n == 0:
            return np.zeros(0, dtype=np.float32)
        query_tokens = tokenize_text(query)
        scores = np.zeros(self.n, dtype=np.float32)
        for i, tf in enumerate(self.term_freqs):
            dl = float(self.lengths[i])
            denom_norm = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-8))
            total = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                total += idf * (freq * (self.k1 + 1.0)) / (freq + denom_norm)
            scores[i] = total
        return scores


def rank_positions(scores):
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


class HybridRetriever:
    def __init__(self, taxonomy, llm, embedding_model, use_embedding=True, batch_size=64):
        self.taxonomy = taxonomy
        self.llm = llm
        self.embedding_model = embedding_model
        self.use_embedding = use_embedding
        self.batch_size = batch_size
        self.embedding_matrix = None
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", embedding_model)
        self.cache_path = taxonomy.csv_path.with_suffix(f".embeddings.{safe_model}.npz")

    def _texts(self):
        if "retrieval_text" in self.taxonomy.df.columns:
            return self.taxonomy.df["retrieval_text"].astype(str).tolist()
        return [
            f"問題：{q}\n答案：{a}"
            for q, a in zip(self.taxonomy.df["question"], self.taxonomy.df["answer"])
        ]

    def ensure_embeddings(self):
        if not self.use_embedding:
            return
        if "retrieval_id" in self.taxonomy.df.columns:
            ids = self.taxonomy.df["retrieval_id"].astype(str).tolist()
        else:
            ids = self.taxonomy.df["atomic_id"].astype(str).tolist()
        if self.embedding_matrix is not None:
            return
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            cached_ids = data["ids"].astype(str).tolist()
            cached_model = str(data["model"].item())
            if cached_ids == ids and cached_model == self.embedding_model:
                self.embedding_matrix = data["vectors"].astype(np.float32)
                return
        texts = self._texts()
        batches = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            print(f"建立 Ollama embedding index：{end}/{total}", file=sys.stderr)
            vectors = self.llm.embed(self.embedding_model, texts[start:end])
            batches.append(vectors)
        matrix = np.vstack(batches).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        np.savez_compressed(
            self.cache_path,
            ids=np.asarray(ids, dtype=str),
            vectors=matrix,
            model=np.asarray(self.embedding_model),
        )
        self.embedding_matrix = matrix

    def query_embedding(self, query):
        self.ensure_embeddings()
        vector = self.llm.embed(self.embedding_model, [query])[0].astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return vector

    def search(self, query, indices, top_k=8, query_vector=None):
        indices = list(dict.fromkeys(int(x) for x in indices))
        if not indices:
            return []
        if "retrieval_text" in self.taxonomy.df.columns:
            texts = [str(self.taxonomy.df.at[idx, "retrieval_text"]) for idx in indices]
        else:
            texts = [
                f"問題：{self.taxonomy.df.at[idx, 'question']}\n答案：{self.taxonomy.df.at[idx, 'answer']}"
                for idx in indices
            ]
        bm25 = BM25(texts)
        bm25_scores = bm25.scores(query)
        bm25_ranks = rank_positions(bm25_scores)
        dense_scores = np.zeros(len(indices), dtype=np.float32)
        if self.use_embedding:
            self.ensure_embeddings()
            if query_vector is None:
                query_vector = self.query_embedding(query)
            dense_scores = self.embedding_matrix[np.asarray(indices)] @ query_vector
            dense_ranks = rank_positions(dense_scores)
            rrf = 1.0 / (60.0 + bm25_ranks) + 1.0 / (60.0 + dense_ranks)
        else:
            rrf = 1.0 / (60.0 + bm25_ranks)
        order = np.argsort(-rrf, kind="stable")[: min(top_k, len(indices))]
        result = []
        for pos in order:
            result.append(
                {
                    "idx": indices[int(pos)],
                    "rrf": float(rrf[int(pos)]),
                    "bm25": float(bm25_scores[int(pos)]),
                    "dense": float(dense_scores[int(pos)]),
                }
            )
        return result
