# app/ranking/rrf.py

from collections import defaultdict


def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=60
):

    scores = defaultdict(float)
    chunk_lookup = {}

    for rank, chunk in enumerate(dense_results):

        uid = (
            chunk["source"],
            chunk["chunk_index"]
        )

        scores[uid] += 1 / (k + rank + 1)

        chunk_lookup[uid] = chunk

    for rank, chunk in enumerate(sparse_results):

        uid = (
            chunk["source"],
            chunk["chunk_index"]
        )

        scores[uid] += 1 / (k + rank + 1)

        chunk_lookup[uid] = chunk

    final_results = []

    for uid, score in scores.items():

        chunk = chunk_lookup[uid].copy()

        chunk["fusion_score"] = score

        final_results.append(chunk)

    final_results.sort(
        key=lambda x: x["fusion_score"],
        reverse=True
    )

    return final_results