import random
from pathlib import Path
import pandas as pd
from src_rag import evaluate, models


N_FILMS = 100          # 100 pour step 1 (None pour tout le corpus)
SEED = 42              
CONFIG = {"model": {"chunk_size": 512, "overlap": 32, "use_reranker": True}}

WIKI = Path("data/raw/movies/wiki")
QUESTIONS_5 = "data/raw/movies/questions.csv"          #separateur ;
QUESTIONS_LONG = "data/raw/movies/questions_long.csv"  #separateur ,
QUESTIONS_TEST = "data/raw/movies/questions_test.csv"  #separateur ;

FILMS_5 = ["Inception.md", "The_Dark_Knight.md", "Deadpool.md", "Fight_Club.md", "Pulp_Fiction.md"]


def choisir_films():
    tous = sorted(p.name for p in WIKI.glob("*.md"))
    obligatoires = [f for f in FILMS_5 if f in tous]

    if N_FILMS is None:
        choisis = tous                      
    else:
        autres = [f for f in tous if f not in obligatoires]
        random.Random(SEED).shuffle(autres)
        n_complement = max(0, N_FILMS - len(obligatoires))
        choisis = obligatoires + autres[:n_complement]

    return [WIKI / name for name in choisis]


def main():
    filenames = choisir_films()
    print(f"Films ingeres : {len(filenames)}")

    rag = models.get_model(CONFIG)
    rag.load_files(filenames)
    print(f"Chunks totaux : {len(rag.get_chunks())}")

    df5 = pd.read_csv(QUESTIONS_5, sep=";").dropna()
    score5 = evaluate.evaluate_retrieval(rag, filenames, df5)
    print(f"MRR (questions 5 films) : {score5['mrr']:.4f}")

    df_long = pd.read_csv(QUESTIONS_LONG, sep=",").dropna()
    score_long = evaluate.evaluate_retrieval(rag, filenames, df_long)
    print(f"MRR (questions longues) : {score_long['mrr']:.4f}")

    return rag


if __name__ == "__main__":
    main()