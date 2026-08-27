import math
import re
import unicodedata
from collections import Counter


class CandidateOrderer:
    """Cheap, local ordering for progressive evidence search.

    Important: this class NEVER removes candidates. It only decides which
    candidate should be examined first. If evidence is still insufficient,
    the caller can continue with later batches until the whole node/frontier
    has been inspected.

    The scorer is intentionally dependency-free. It uses a small BM25-like
    lexical score over English/alphanumeric tokens, MARC-ish field/position
    tokens, and Chinese character n-grams. This avoids an extra embedding or
    LLM call in the online path.
    """

    _STOPWORDS = {
        "請問",
        "老師",
        "您好",
        "想了解",
        "想請教",
        "可以",
        "是否",
        "有何",
        "什麼",
        "一下",
        "謝謝",
        "如何",
        "以及",
        "另外",
        "the",
        "an",
        "of",
        "is",
        "are",
        "what",
        "which",
        "的",
        "與",
        "和",
    }

    def __init__(self, k1=1.5, b=0.75, top_units_for_pack_score=3):
        self.k1 = float(k1)
        self.b = float(b)
        self.top_units_for_pack_score = max(1, int(top_units_for_pack_score))

    @staticmethod
    def _normalize(text):
        return unicodedata.normalize("NFKC", str(text or "")).lower()

    def _tokens(self, text):
        text = self._normalize(text)
        tokens = []

        # MARC-style location tokens, e.g. Leader/07 or LDR/06.
        for match in re.finditer(r"\b(?:leader|ldr)\s*/?\s*\d{1,2}\b", text):
            value = re.sub(r"\s+", "", match.group())
            tokens.append(value)

        # Subfields such as $h, $a, $1.
        tokens.extend(re.findall(r"\$[0-9a-z]", text))

        # Preserve single-letter code values (a, m, s...) because they can be
        # semantically critical in MARC/CMARC questions.
        tokens.extend(re.findall(r"[a-z]+|\d+", text))

        # Chinese phrases are represented by overlapping 2- and 3-grams.
        for seq in re.findall(r"[\u3400-\u9fff]+", text):
            if len(seq) <= 8:
                tokens.append(seq)
            for n in (2, 3):
                if len(seq) >= n:
                    tokens.extend(seq[i : i + n] for i in range(len(seq) - n + 1))

        return [
            token
            for token in tokens
            if token and token not in self._STOPWORDS
        ]

    def _bm25_scores(self, query_text, documents):
        documents = [str(doc or "") for doc in documents]
        if not documents:
            return []

        query_tokens = self._tokens(query_text)
        doc_tokens = [self._tokens(doc) for doc in documents]
        if not query_tokens:
            return [0.0] * len(documents)

        n_docs = len(doc_tokens)
        avgdl = sum(len(tokens) for tokens in doc_tokens) / max(1, n_docs)
        avgdl = max(avgdl, 1.0)

        df = Counter()
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] += 1

        query_tf = Counter(query_tokens)
        scores = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            dl = max(1, len(tokens))
            score = 0.0
            for token, qfreq in query_tf.items():
                freq = tf.get(token, 0)
                if not freq:
                    continue

                idf = math.log(
                    1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5)
                )
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (dl / avgdl)
                )
                score += (
                    idf
                    * (freq * (self.k1 + 1.0) / denom)
                    * (1.0 + 0.12 * min(qfreq - 1, 4))
                )

            scores.append(float(score))
        return scores

    @staticmethod
    def _unit_text(unit):
        return "\n".join(
            [
                str(unit.get("content", "")),
                " ".join(str(x) for x in unit.get("source_ids", [])),
            ]
        )

    def order_units(self, query, units, extra_text=""):
        """Return every unit, reordered by cheap lexical relevance."""
        units = list(units or [])
        if len(units) <= 1:
            return units

        focus = f"{query}\n{extra_text}".strip()
        docs = [self._unit_text(unit) for unit in units]
        scores = self._bm25_scores(focus, docs)

        ranked = list(enumerate(units))
        ranked.sort(
            key=lambda pair: (
                -scores[pair[0]],
                pair[0],
            )
        )
        return [unit for _, unit in ranked]

    def score_pack(self, query, pack, extra_text=""):
        """Score a node/frontier pack without deleting any unit."""
        units = list(pack.get("knowledge_units", []) or [])
        if not units:
            return 0.0

        focus = f"{query}\n{extra_text}".strip()
        docs = [self._unit_text(unit) for unit in units]
        unit_scores = sorted(self._bm25_scores(focus, docs), reverse=True)
        top = unit_scores[: self.top_units_for_pack_score]
        top_mean = sum(top) / max(1, len(top))
        top_max = top[0] if top else 0.0

        # Node/path wording provides a light additional signal only.
        path_score = self._bm25_scores(focus, [str(pack.get("path", ""))])[0]
        return float(top_mean + 0.35 * top_max + 0.20 * path_score)


    def score_packs_global(self, query, packs, extra_text=""):
        """Comparable pack scores computed from one shared candidate corpus.

        Unlike score_pack(), this uses a single BM25 document-frequency space
        across every supplied pack, which is better for choosing the next node
        from a heterogeneous progressive frontier.
        """
        packs = list(packs or [])
        if not packs:
            return []

        focus = f"{query}\n{extra_text}".strip()
        documents = []
        owners = []
        for pack_index, pack in enumerate(packs):
            for unit in pack.get("knowledge_units", []) or []:
                documents.append(self._unit_text(unit))
                owners.append(pack_index)

        if not documents:
            return [0.0] * len(packs)

        scores = self._bm25_scores(focus, documents)
        by_pack = [[] for _ in packs]
        for owner, score in zip(owners, scores):
            by_pack[owner].append(float(score))

        output = []
        for pack, values in zip(packs, by_pack):
            values.sort(reverse=True)
            top = values[: self.top_units_for_pack_score]
            top_mean = sum(top) / max(1, len(top)) if top else 0.0
            top_max = top[0] if top else 0.0
            path_score = self._bm25_scores(
                focus,
                [str(pack.get("path", ""))],
            )[0]
            output.append(
                float(top_mean + 0.35 * top_max + 0.20 * path_score)
            )
        return output
    def rank_packs(self, query, packs, extra_text=""):
        """Return all packs ordered by local relevance; no hard pruning."""
        scored = [
            (self.score_pack(query, pack, extra_text=extra_text), i, pack)
            for i, pack in enumerate(packs or [])
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"pack": pack, "local_score": float(score)}
            for score, _, pack in scored
        ]
