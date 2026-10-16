import os
import re

import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv

from src_rag import embedding


load_dotenv()

tokenizer = tiktoken.get_encoding("cl100k_base")

EMBEDDER_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDER_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_model_name(provider=None):
    provider = provider or os.environ["LLM_PROVIDER"]
    return {
        "GROQ": "openai/gpt-oss-20b",
        "OPEN_ROUTER": "openai/gpt-oss-20b",
    }[provider]


def get_client(provider=None):
    provider = provider or os.environ["LLM_PROVIDER"]
    url = {
        "GROQ": "https://api.groq.com/openai/v1",
        "OPEN_ROUTER": "https://openrouter.ai/api/v1",
    }[provider]

    return openai.OpenAI(
        base_url=url,
        api_key=os.environ[f"{provider}_API_KEY"],
    )


def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(
        self,
        chunk_size=256,
        overlap=32,
        add_headers=True,
        provider=None,
        use_reranker=False,
        reranker_model="BAAI/bge-reranker-base",
        use_bm25=False,
        rrf_k=60,
        top_k=5,
        pool=30,
    ):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._add_headers = add_headers
        self._top_k = top_k
        self._pool = pool

        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._corpus_embedding = None

        self._provider = provider or os.environ["LLM_PROVIDER"]
        self._client = get_client(self._provider)
        self._model_name = get_model_name(self._provider)

        self._use_reranker = use_reranker
        self._reranker = None
        if use_reranker:
            from FlagEmbedding import FlagReranker
            self._reranker = FlagReranker(reranker_model, use_fp16=True)

        self._use_bm25 = use_bm25
        self._rrf_k = rrf_k
        self._bm25 = None

    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename) as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        if not texts:
            return  

        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

        if self._use_bm25:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([self._tokenize(c) for c in self._chunks])

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        return self._embed([EMBEDDER_QUERY_INSTRUCTION + q for q in questions])

    def embed_corpus(self, chunks):
        return self._embed(chunks)

    def _compute_chunks(self, texts):
        return sum(
            (
                chunk_markdown(
                    txt,
                    chunk_size=self._chunk_size,
                    overlap=self._overlap,
                    add_headers=self._add_headers,
                )
                for txt in texts
            ),
            [],
        )

    def _embed(self, texts):
        if not texts:
            return np.empty((0, 0))
        return embedding.embed(texts, EMBEDDER_MODEL, use_fp16=True)

    def reply(self, query):
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_name,
        )
        return res.choices[0].message.content

    def _build_prompt(self, query):
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply \"I cannot answer that question\".
Query: {query}
Answer:"""

    def _get_context(self, query):
        q_emb = self.embed_questions([query])
        sims = (q_emb @ self._corpus_embedding.T)[0]

        want = self._pool if self._use_reranker else self._top_k

        if self._use_bm25:
            dense_order = list(np.argsort(sims)[::-1][:100])
            bm25_scores = self._bm25.get_scores(self._tokenize(query))
            bm25_order = list(np.argsort(bm25_scores)[::-1][:100])
            cand = self._rrf([dense_order, bm25_order], k=self._rrf_k)[:want]
        else:
            cand = list(np.argsort(sims)[::-1][:want])

        if self._use_reranker:
            pairs = [[query, self._chunks[i]] for i in cand]
            scores = self._reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
            cand = [i for _, i in sorted(zip(scores, cand), key=lambda t: t[0], reverse=True)]

        return [self._chunks[i] for i in cand[: self._top_k]]

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _rrf(rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                idx = int(idx)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return [idx for idx, _ in sorted(scores.items(), key=lambda t: t[1], reverse=True)]


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    """
    Parses markdown into a list of {'headers': [...], 'content': ...}
    Preserves full header hierarchy (e.g. ["Section", "Sub", "SubSub", ...])
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            if current_section["content"]:
                sections.append(current_section)

            header_stack = header_stack[: level - 1]
            header_stack.append(title)

            current_section = {"headers": header_stack.copy(), "content": ""}
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(
    md_text: str,
    chunk_size: int = 256,
    overlap: int = 32,
    add_headers: bool = True,
) -> list[str]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []
    step = max(1, chunk_size - overlap)

    for section in parsed_sections:
        prefix = " > ".join(section["headers"]) if add_headers else ""
        tokens = tokenizer.encode(section["content"])

        for i in range(0, len(tokens), step):
            piece = tokens[i : i + chunk_size]
            if not piece:
                continue
            text = tokenizer.decode(piece)
            chunks.append(f"{prefix}\n{text}" if prefix else text)

    return chunks