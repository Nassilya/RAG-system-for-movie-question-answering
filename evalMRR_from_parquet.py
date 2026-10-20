import numpy as np
import pandas as pd

from src_rag import models


CONFIG = {"model": {"use_reranker": True}}   
CHUNK_PARQUET = "chunk.parquet"
QUESTIONS_5 = "data/raw/movies/questions.csv"          
QUESTIONS_LONG = "data/raw/movies/questions_long.csv"  


def charger_rag_depuis_parquet():
    df = pd.read_parquet(CHUNK_PARQUET)
    chunks = df["chunk"].tolist()
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

    rag = models.get_model(CONFIG)
    rag._chunks = chunks                 
    rag._corpus_embedding = embeddings   
    print(f"Charge depuis {CHUNK_PARQUET} : {len(chunks)} chunks, embeddings {embeddings.shape}")
    return rag


def mrr_sur(rag, chemin_csv, sep):
    df = pd.read_csv(chemin_csv, sep=sep).dropna(subset=["question", "text_answering"])

    ranks = []
    for _, row in df.iterrows():
        chunks = rag._get_context(row["question"])
        try:
            rank = next(i + 1 for i, c in enumerate(chunks) if row["text_answering"] in c)
        except StopIteration:
            rank = 0
        ranks.append(rank)

    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])
    return mrr, len(df)


def main():
    rag = charger_rag_depuis_parquet()

    mrr5, n5 = mrr_sur(rag, QUESTIONS_5, ";")
    print(f"MRR questions.csv (5 films, {n5} questions)        : {mrr5:.4f}")

    mrr_long, nl = mrr_sur(rag, QUESTIONS_LONG, ",")
    print(f"MRR questions_long.csv (tous films, {nl} questions) : {mrr_long:.4f}")


if __name__ == "__main__":
    main()