
import pandas as pd
from src_rag import evaluate, models


CONFIG = {"model": {"chunk_size": 512, "overlap": 32, "use_reranker": True}}
QUESTIONS_TEST = "data/raw/movies/questions_test.csv"


def main():
    rag = models.get_model(CONFIG)
    rag.load_files(evaluate.FILENAMES)

    
    emb = rag.get_corpus_embedding()
    pd.DataFrame(
        {
            "chunk": rag.get_chunks(),
            "embedding": [e.tolist() for e in emb],
        }
    ).to_parquet("chunk.parquet")

    
    test = pd.read_csv(QUESTIONS_TEST, sep=";")
    test["embedding"] = [rag.embed_questions([q])[0].tolist() for q in test["question"]]
    test["reply"] = [rag.reply(q) for q in test["question"]]
    test.to_parquet("question.parquet")

    print(f"chunk.parquet : {len(rag.get_chunks())} chunks")
    print(f"question.parquet : {len(test)} questions")


if __name__ == "__main__":
    main()