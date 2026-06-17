# app/retrievers/bm25_retriever.py

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):
        """
        chunks:
        [
            {
                "text": "...",
                "source": "...",
                "chunk_index": 0
            }
        ]
        """

        self.chunks = chunks

        corpus = [
            chunk["text"].split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for idx in ranked_indices:

            chunk = self.chunks[idx].copy()

            chunk["score"] = float(scores[idx])

            results.append(chunk)

        return results