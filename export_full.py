"""
    question.parquet appelle le LLM pour chaque question de test:
    
    prevoir une cle API Groq/OpenRouter valide dans le .env


"""

from pathlib import Path

import pandas as pd

from src_rag import models


CONFIG = {"model": {"chunk_size": 512, "overlap": 32, "use_reranker": True}}

WIKI = Path("data/raw/movies/wiki")
QUESTIONS_TEST = "data/raw/movies/questions_test.csv"  #separateur ;


def main():
    filenames = sorted(WIKI.glob("*.md"))
    print(f"Films a ingerer : {len(filenames)}")

    rag = models.get_model(CONFIG)
    rag.load_files(filenames)   
    print(f"Chunks totaux : {len(rag.get_chunks())}")

   
    emb = rag.get_corpus_embedding()
    pd.DataFrame(
        {
            "chunk": rag.get_chunks(),
            "embedding": [e.tolist() for e in emb],
        }
    ).to_parquet("chunk.parquet")
    print(f"chunk.parquet ecrit : {len(rag.get_chunks())} chunks")

    
    test = pd.read_csv(QUESTIONS_TEST, sep=";")
    test["embedding"] = [rag.embed_questions([q])[0].tolist() for q in test["question"]]
    test["reply"] = [rag.reply(q) for q in test["question"]]
    test.to_parquet("question.parquet")
    print(f"question.parquet ecrit : {len(test)} questions")


if __name__ == "__main__":
    main()