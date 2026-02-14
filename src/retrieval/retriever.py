"""
Retriever
==========
Orchestrates the query flow: takes a natural-language query, encodes it,
searches the FAISS index, and returns ranked context chunks.

This is the "R" in RAG.  The retriever does NOT generate answers -- it
only finds relevant context.  Answer generation is handled by the LLM
module (src/llm/ollama_client.py).

Pipeline (basic):
  1. User query (string)
  2. Encode query -> dense vector (via EmbeddingEncoder)
  3. Search FAISS index -> top-k chunk metadata + scores
  4. Return ranked list of context chunks

Extended pipeline (with TODOs implemented):
  1. Preprocess query  (TODO 2: normalise, correct spelling, expand)
  2. Dense retrieval   (encode -> FAISS search)
  3. Sparse retrieval  (TODO 3: BM25 keyword match)
  4. Merge & dedupe    (TODO 3: combine dense + sparse via RRF)
  5. Metadata filter   (TODO 4: keep only results matching filters)
  6. Re-rank           (TODO 1: cross-encoder re-scoring)
  7. Format context    (for LLM prompt construction)

Design decisions:
  - Retrieval is embedding-based, NOT LLM-based.  The LLM is only used
    for final answer generation after context is retrieved.
  - The retriever is stateless -- it receives dependencies via constructor.
  - All new features (cross-encoder, BM25, etc.) are optional --
    they degrade gracefully when the required packages are missing.

Learning TODO (all implemented):
  1. Implement re-ranking using cross-encoder models.
  2. Add query preprocessing (spelling correction, expansion).
  3. Add hybrid retrieval (combine dense + sparse / BM25).
  4. Add metadata filtering (e.g., search only PDFs, or only
        a specific dataset).
"""

import logging
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Set

import numpy as np

from src.embeddings.encoder import EmbeddingEncoder
from src.index.faiss_index import FaissIndex

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Optional dependency: cross-encoder for re-ranking (TODO 1)
# ------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class RetrieverResult:
    """
    A single retrieval result with score and chunk information.
    """

    def __init__(self, chunk_id: str, doc_id: str, text: str, score: float, metadata: Dict):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"RetrieverResult(score={self.score:.4f}, chunk='{preview}...')"

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


# ======================================================================
#  TODO 2 — Query Preprocessing
# ======================================================================

class QueryPreprocessor:
    """
    Cleans and optionally expands a raw user query before retrieval.

    Why preprocess?
    ---------------
    Users type messy queries: extra spaces, mixed case, typos, and
    abbreviations.  Even though embedding models are somewhat robust
    to these, small normalisation steps can measurably improve recall:

    - **Normalisation** — collapse whitespace, strip punctuation
      clutter, and lowercase the query so the tokeniser sees
      a cleaner input.
    - **Spelling correction** — swap common misspellings for their
      correct forms.  We use a simple dictionary here; in production
      you would use a library like `pyspellchecker` or `TextBlob`.
    - **Synonym expansion** — append synonyms to the query so the
      embedding captures related concepts.  For example, searching
      for "invoice" also surfaces chunks that say "bill" or "receipt".

    How it fits the pipeline
    ------------------------
    Preprocessing happens *before* the query is embedded.  The idea is
    to give the embedding model the best possible input so it produces
    a vector that truly represents the user's intent.

    ``raw query  →  preprocess()  →  clean query  →  encode  →  vector``

    Design notes
    ------------
    - This is intentionally simple.  No external NLP models are loaded.
    - The synonym map is small and domain-specific.  Extend it for
      your own document collection.
    - The spelling dictionary covers only very common mistakes.
      For serious spelling correction, plug in ``pyspellchecker``.
    """

    # Common misspellings -> correct form
    # (extend this for your domain)
    SPELLING_FIXES: Dict[str, str] = {
        "invoce":    "invoice",
        "invioce":   "invoice",
        "reciept":   "receipt",
        "recieve":   "receive",
        "adress":    "address",
        "ammount":   "amount",
        "amoutn":    "amount",
        "signiture": "signature",
        "employe":   "employee",
        "documnet":  "document",
        "infomation":"information",
        "sumary":    "summary",
        "purchse":   "purchase",
        "compnay":   "company",
        "accout":    "account",
        "totla":     "total",
        "fomr":      "form",
    }

    # Synonyms: if any key appears in the query, append the values
    # so the embedding captures related concepts.
    SYNONYMS: Dict[str, List[str]] = {
        "invoice":  ["bill", "receipt", "statement"],
        "employee": ["worker", "staff", "personnel"],
        "amount":   ["total", "sum", "value", "price"],
        "address":  ["location", "place"],
        "company":  ["organisation", "firm", "business"],
        "date":     ["day", "time", "when"],
        "sign":     ["signature", "signed", "autograph"],
        "name":     ["person", "who", "identity"],
        "purchase": ["buy", "order", "procurement"],
    }

    def preprocess(
        self,
        query: str,
        fix_spelling: bool = True,
        expand_synonyms: bool = True,
    ) -> str:
        """
        Clean and optionally expand a query string.

        Args:
            query           : raw user query
            fix_spelling    : apply dictionary-based spelling fixes
            expand_synonyms : append synonyms for key terms

        Returns:
            Preprocessed query string.
        """
        # 1. Basic normalisation
        text = query.strip()
        text = re.sub(r"\s+", " ", text)           # collapse whitespace
        text = text.lower()

        # 2. Spelling correction (word-level dictionary lookup)
        if fix_spelling:
            words = text.split()
            words = [self.SPELLING_FIXES.get(w, w) for w in words]
            text = " ".join(words)

        # 3. Synonym expansion
        if expand_synonyms:
            expansions: List[str] = []
            for key, syns in self.SYNONYMS.items():
                if key in text:
                    expansions.extend(syns)
            if expansions:
                text = text + " " + " ".join(expansions)

        logger.debug("Query preprocessed: '%s' -> '%s'", query, text)
        return text


# ======================================================================
#  TODO 3 — BM25 Sparse Retriever (keyword-based)
# ======================================================================

class BM25Retriever:
    """
    A pure-Python BM25 (Best Matching 25) implementation for sparse
    keyword-based retrieval.

    What is BM25?
    -------------
    BM25 is a **term-frequency / inverse-document-frequency** scoring
    function.  It was the state-of-the-art in information retrieval
    for decades before neural embeddings took over.

    Unlike dense retrieval (embeddings + FAISS), BM25 works on **exact
    word overlap**.  If the user searches for "invoice", BM25 will find
    chunks that literally contain the word "invoice" — even if the
    embedding model would assign a low semantic similarity.

    The scoring formula
    -------------------
    For each query term ``q`` in query ``Q`` and document ``D``::

        score(D, Q) = Σ  IDF(q) · [ f(q,D) · (k1 + 1) ]
                                    ───────────────────────
                                    f(q,D) + k1 · (1 - b + b · |D|/avgdl)

    Where:
    - ``f(q, D)``  = frequency of term ``q`` in document ``D``
    - ``|D|``      = length of document ``D`` (in words)
    - ``avgdl``    = average document length across the corpus
    - ``IDF(q)``   = inverse document frequency of term ``q``
    - ``k1``       = term frequency saturation parameter (default 1.5)
    - ``b``        = length normalisation parameter (default 0.75)

    Key intuitions:
    - **IDF** — rare words matter more.  "invoice" is more informative
      than "the".
    - **TF saturation** — seeing a word 10 times isn't 10× better than
      seeing it once.  ``k1`` controls how quickly the benefit saturates.
    - **Length normalisation** — longer documents naturally contain more
      words, so we normalise by ``|D| / avgdl``.  ``b`` controls how
      aggressively we penalise long documents.

    Why combine with dense retrieval?
    ---------------------------------
    Dense retrieval (FAISS) is great at **semantic** matching — it finds
    chunks that *mean* the same thing even if they use different words.
    But it can miss chunks that contain the **exact keyword** the user
    typed (especially for proper nouns, codes, or rare terms).

    BM25 fills this gap.  Combining both is called **hybrid retrieval**
    and typically outperforms either approach alone.

    How they are combined (Reciprocal Rank Fusion)
    -----------------------------------------------
    See the ``_reciprocal_rank_fusion()`` method in ``Retriever``.

    Implementation notes
    --------------------
    This is a self-contained, dependency-free implementation.  For
    production use at scale, consider ``rank-bm25``, ``Pyserini``, or
    Elasticsearch.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1 : term frequency saturation.  Higher = diminishing returns
                 kick in later.  Typical range: 1.2 – 2.0.
            b  : length normalisation.  0 = no normalisation,
                 1 = full normalisation.  Typical: 0.75.
        """
        self.k1 = k1
        self.b = b

        # Populated by fit()
        self._corpus_tokens: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        self._df: Dict[str, int] = {}       # document frequency per term
        self._idf: Dict[str, float] = {}    # precomputed IDF per term
        self._metadata: List[Dict] = []     # parallel metadata

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Simple whitespace + punctuation tokeniser."""
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: List[Dict]) -> None:
        """
        Index a list of document dicts (must have a ``"text"`` key).

        This computes IDF values and stores tokenised documents for
        scoring at query time.

        Args:
            documents : list of dicts, each with at least a ``"text"`` key.
                        Other keys are preserved as metadata.
        """
        self._metadata = documents
        self._corpus_tokens = []
        self._df = defaultdict(int)

        for doc in documents:
            tokens = self._tokenise(doc.get("text", ""))
            self._corpus_tokens.append(tokens)
            # Document frequency — count each unique term once per doc
            for term in set(tokens):
                self._df[term] += 1

        self._n_docs = len(documents)
        self._doc_lengths = [len(t) for t in self._corpus_tokens]
        self._avgdl = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs else 1.0
        )

        # Precompute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df_val in self._df.items():
            self._idf[term] = math.log(
                (self._n_docs - df_val + 0.5) / (df_val + 0.5) + 1.0
            )

        logger.info(
            "BM25 fitted: %d documents, %d unique terms, avgdl=%.1f",
            self._n_docs,
            len(self._df),
            self._avgdl,
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Score all documents against the query and return the top-k.

        Args:
            query : raw query string
            top_k : number of results to return

        Returns:
            List of dicts (copies of metadata + ``"score"`` key),
            sorted by descending BM25 score.
        """
        query_tokens = self._tokenise(query)
        if not query_tokens or not self._corpus_tokens:
            return []

        scores = np.zeros(self._n_docs, dtype=np.float64)

        for qt in query_tokens:
            idf = self._idf.get(qt, 0.0)
            if idf == 0.0:
                continue  # term not in corpus

            for i, doc_tokens in enumerate(self._corpus_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = self._doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self._avgdl
                )
                scores[i] += idf * (numerator / denominator)

        # Get top-k indices
        top_k = min(top_k, self._n_docs)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break  # no point returning zero-score results
            entry = {**self._metadata[idx], "score": float(scores[idx])}
            results.append(entry)

        return results


# ======================================================================
#  TODO 1 — Cross-Encoder Re-ranker
# ======================================================================

class CrossEncoderReranker:
    """
    Re-ranks retrieval results using a cross-encoder model.

    What is a cross-encoder?
    ------------------------
    An embedding model (bi-encoder) encodes the query and document
    **independently** — they never see each other.  This is fast
    (encode once, compare with dot product) but loses fine-grained
    interaction between query words and document words.

    A cross-encoder takes the query and document **together** as a
    single input::

        [CLS] query text [SEP] document text [SEP]  →  relevance score

    The Transformer's self-attention can now directly compare every
    query token against every document token.  This produces much
    more accurate relevance scores, but is **expensive** — you must
    run a full forward pass for every (query, document) pair.

    Why re-rank instead of using cross-encoder for retrieval?
    ---------------------------------------------------------
    Cross-encoders are O(N) per query where N is the number of
    candidate documents.  For a corpus of 100k chunks, that's 100k
    forward passes — wildly impractical.

    The solution is a **two-stage pipeline**:
    1. **Stage 1 (recall)** — Use a fast bi-encoder + FAISS to
       retrieve a candidate set (e.g. top-50).
    2. **Stage 2 (precision)** — Use the cross-encoder to re-score
       only those 50 candidates and re-order them.

    This gives you the speed of bi-encoders with the accuracy of
    cross-encoders.

    Model used
    ----------
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — a small (80 MB),
    fast cross-encoder trained on the MS MARCO passage ranking
    dataset.  It outputs a single relevance score in [-∞, +∞]
    (higher = more relevant).

    Requirements
    ------------
    ``pip install sentence-transformers`` (the CrossEncoder class
    lives in the sentence-transformers library).
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        """
        Args:
            model_name : HuggingFace model identifier for the cross-encoder
            device     : "cpu" or "cuda"
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for cross-encoder "
                "re-ranking.  Install: pip install sentence-transformers"
            )
        logger.info("Loading cross-encoder: %s on %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List["RetrieverResult"],
        top_k: Optional[int] = None,
    ) -> List["RetrieverResult"]:
        """
        Re-score and re-sort results using the cross-encoder.

        Args:
            query   : the original query string
            results : list of RetrieverResult from the first-stage retriever
            top_k   : if set, return only the top-k after re-ranking

        Returns:
            New list of RetrieverResult objects, sorted by cross-encoder
            score (descending).  The ``score`` field is *replaced* with
            the cross-encoder score.
        """
        if not results:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, r.text) for r in results]

        # Score all pairs in one batch
        ce_scores = self.model.predict(pairs)

        # Attach new scores and sort
        scored: List[tuple] = sorted(
            zip(ce_scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked: List[RetrieverResult] = []
        for score, result in scored:
            new_result = RetrieverResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                score=float(score),
                metadata={**result.metadata, "original_score": result.score},
            )
            reranked.append(new_result)

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(
            "Cross-encoder re-ranked %d -> %d results",
            len(results),
            len(reranked),
        )
        return reranked


# ======================================================================
#  Main Retriever class (extended with all TODOs)
# ======================================================================

class Retriever:
    """
    Embedding-based retriever over chunked documents.

    Now supports:
      - Query preprocessing (TODO 2)
      - Dense retrieval via FAISS (original)
      - Sparse retrieval via BM25 (TODO 3)
      - Hybrid fusion of dense + sparse (TODO 3)
      - Metadata filtering (TODO 4)
      - Cross-encoder re-ranking (TODO 1)

    Usage (basic — same as before):
        encoder = EmbeddingEncoder()
        index = FaissIndex(dimension=384)
        index.load("data/processed/index")

        retriever = Retriever(encoder=encoder, index=index)
        results = retriever.query("What is the invoice total?", top_k=5)

    Usage (with all features):
        retriever = Retriever(
            encoder=encoder,
            index=index,
            enable_bm25=True,       # hybrid retrieval
            enable_reranker=True,    # cross-encoder stage 2
            enable_preprocessing=True,
        )
        results = retriever.query(
            "invoce total ammount",  # typos get corrected!
            top_k=5,
            filters={"doc_type": "form"},  # only forms
        )
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: FaissIndex,
        enable_bm25: bool = False,
        enable_reranker: bool = False,
        enable_preprocessing: bool = False,
        reranker_model: str = CrossEncoderReranker.DEFAULT_MODEL,
        reranker_device: str = "cpu",
    ):
        """
        Args:
            encoder               : embedding encoder for dense retrieval
            index                 : FAISS index (already loaded/built)
            enable_bm25           : activate BM25 hybrid retrieval
            enable_reranker       : activate cross-encoder re-ranking
            enable_preprocessing  : activate query spelling/synonym expansion
            reranker_model        : HuggingFace model for cross-encoder
            reranker_device       : "cpu" or "cuda" for cross-encoder
        """
        self.encoder = encoder
        self.index = index

        # TODO 2: query preprocessor
        self._preprocessor: Optional[QueryPreprocessor] = None
        if enable_preprocessing:
            self._preprocessor = QueryPreprocessor()
            logger.info("Query preprocessing enabled")

        # TODO 3: BM25 sparse retriever
        self._bm25: Optional[BM25Retriever] = None
        if enable_bm25 and index.metadata:
            self._bm25 = BM25Retriever()
            self._bm25.fit(index.metadata)
            logger.info("BM25 hybrid retrieval enabled (%d docs)", len(index.metadata))

        # TODO 1: cross-encoder re-ranker
        self._reranker: Optional[CrossEncoderReranker] = None
        if enable_reranker:
            self._reranker = CrossEncoderReranker(
                model_name=reranker_model,
                device=reranker_device,
            )
            logger.info("Cross-encoder re-ranking enabled")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, str]] = None,
        use_hybrid: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
        use_preprocessing: Optional[bool] = None,
    ) -> List[RetrieverResult]:
        """
        Retrieve the most relevant chunks for a natural-language query.

        The full pipeline is::

            preprocess → dense search → (+ BM25 → fuse) → filter → re-rank

        Each stage is skipped if its feature is disabled or the
        per-call flag is set to False.

        Args:
            query_text         : the user's question or search string
            top_k              : maximum number of results to return
            score_threshold    : discard results below this score
            filters            : metadata key-value pairs to keep
                                 (see TODO 4 explanation below)
            use_hybrid         : per-call override for BM25 fusion
                                 (None = use constructor setting)
            use_reranker       : per-call override for cross-encoder
                                 (None = use constructor setting)
            use_preprocessing  : per-call override for query preprocessing
                                 (None = use constructor setting)

        Returns:
            List of RetrieverResult objects, sorted by descending score.
        """
        if not query_text.strip():
            logger.warning("Empty query, returning no results")
            return []

        # ----------------------------------------------------------
        #  TODO 2: Query preprocessing
        # ----------------------------------------------------------
        do_preprocess = (
            use_preprocessing if use_preprocessing is not None
            else self._preprocessor is not None
        )
        if do_preprocess:
            preprocessor = self._preprocessor or QueryPreprocessor()
            processed_query = preprocessor.preprocess(query_text)
            logger.info(
                "Preprocessed query: '%s' -> '%s'",
                query_text[:60],
                processed_query[:80],
            )
        else:
            processed_query = query_text

        # ----------------------------------------------------------
        #  Dense retrieval (original FAISS path)
        # ----------------------------------------------------------
        # When using hybrid, retrieve more candidates for fusion
        do_hybrid = (
            use_hybrid if use_hybrid is not None
            else self._bm25 is not None
        )
        dense_top_k = top_k * 3 if do_hybrid else top_k

        query_vector = self.encoder.encode_single(processed_query, normalize=True)
        raw_results = self.index.search(query_vector, top_k=dense_top_k)

        # ----------------------------------------------------------
        #  TODO 3: Hybrid retrieval (dense + BM25 + RRF fusion)
        # ----------------------------------------------------------
        if do_hybrid and self._bm25 is not None:
            bm25_results = self._bm25.search(processed_query, top_k=dense_top_k)
            raw_results = self._reciprocal_rank_fusion(
                dense_results=raw_results,
                sparse_results=bm25_results,
                top_k=dense_top_k,
            )
            logger.info(
                "Hybrid fusion: %d dense + %d sparse -> %d merged",
                len(raw_results),
                len(bm25_results),
                len(raw_results),
            )

        # Wrap into RetrieverResult objects
        results = self._wrap_results(raw_results, score_threshold)

        # ----------------------------------------------------------
        #  TODO 4: Metadata filtering
        # ----------------------------------------------------------
        if filters:
            results = self._apply_filters(results, filters)

        # ----------------------------------------------------------
        #  TODO 1: Cross-encoder re-ranking
        # ----------------------------------------------------------
        do_rerank = (
            use_reranker if use_reranker is not None
            else self._reranker is not None
        )
        if do_rerank and self._reranker is not None:
            results = self._reranker.rerank(
                query_text,  # use original query, not preprocessed
                results,
                top_k=top_k,
            )
        else:
            results = results[:top_k]

        logger.info(
            "Query '%s' -> %d results (top_k=%d, threshold=%s, "
            "hybrid=%s, rerank=%s, preprocess=%s)",
            query_text[:60],
            len(results),
            top_k,
            score_threshold,
            do_hybrid,
            do_rerank,
            do_preprocess,
        )
        return results

    # ------------------------------------------------------------------
    #  Internal: wrap raw dicts into RetrieverResult
    # ------------------------------------------------------------------

    def _wrap_results(
        self,
        raw_results: List[Dict],
        score_threshold: Optional[float] = None,
    ) -> List[RetrieverResult]:
        """Convert raw result dicts to RetrieverResult objects."""
        results: List[RetrieverResult] = []
        for entry in raw_results:
            if score_threshold is not None and entry["score"] < score_threshold:
                continue
            result = RetrieverResult(
                chunk_id=entry.get("chunk_id", "unknown"),
                doc_id=entry.get("doc_id", "unknown"),
                text=entry.get("text", ""),
                score=entry["score"],
                metadata={
                    k: v
                    for k, v in entry.items()
                    if k not in ("chunk_id", "doc_id", "text", "score")
                },
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    #  TODO 3: Reciprocal Rank Fusion (RRF)
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int = 20,
        k: int = 60,
    ) -> List[Dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        Why RRF?
        --------
        Dense and sparse scores live on completely different scales
        (cosine similarity ∈ [-1, 1] vs BM25 scores ∈ [0, ∞]).
        You can't simply add or average them.

        RRF avoids this problem by ignoring the *scores* entirely
        and using only the *ranks*.  For each result, its RRF score
        is::

            RRF(d) = Σ  1 / (k + rank_i(d))
                     i

        where the sum is over all ranked lists that contain document
        ``d``, and ``k`` is a constant (typically 60) that prevents
        top-ranked documents from dominating too aggressively.

        Example
        -------
        Suppose document X is ranked #1 in dense and #3 in BM25::

            RRF(X) = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

        Document Y is ranked #5 in dense and #1 in BM25::

            RRF(Y) = 1/(60+5) + 1/(60+1) = 0.01538 + 0.01639 = 0.03177

        So X still ranks above Y, but Y gets a significant boost from
        its strong BM25 rank.

        Args:
            dense_results  : ranked results from FAISS
            sparse_results : ranked results from BM25
            top_k          : how many merged results to return
            k              : RRF smoothing constant (default 60)

        Returns:
            Merged list of result dicts, sorted by RRF score.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Dict] = {}

        # Score dense results by rank
        for rank, entry in enumerate(dense_results, start=1):
            key = entry.get("chunk_id", str(rank))
            rrf_scores[key] += 1.0 / (k + rank)
            doc_map[key] = entry

        # Score sparse results by rank
        for rank, entry in enumerate(sparse_results, start=1):
            key = entry.get("chunk_id", f"bm25_{rank}")
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = entry

        # Sort by fused score
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        merged: List[Dict] = []
        for key in sorted_keys[:top_k]:
            entry = {**doc_map[key], "score": rrf_scores[key]}
            merged.append(entry)

        return merged

    # ------------------------------------------------------------------
    #  TODO 4: Metadata filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_filters(
        results: List[RetrieverResult],
        filters: Dict[str, str],
    ) -> List[RetrieverResult]:
        """
        Keep only results whose metadata matches ALL filter criteria.

        Why filter?
        -----------
        A knowledge base may contain documents from multiple datasets
        (funsd, docvqa, rvl_cdip), multiple document types (form,
        invoice, letter), or multiple sources.  Sometimes the user
        knows they only care about a subset — e.g. "search only
        invoices" or "only funsd documents".

        Metadata filtering happens *after* retrieval but *before*
        re-ranking, so irrelevant results are discarded early.

        How it works
        ------------
        Each filter is a key-value pair.  A result passes if its
        metadata contains that key with a matching value (case-
        insensitive substring match).  ALL filters must pass.

        Examples::

            filters={"doc_type": "form"}
            # keeps only results where metadata["doc_type"] contains "form"

            filters={"source": "funsd", "doc_type": "form"}
            # keeps only results matching BOTH conditions

        Args:
            results : list of RetrieverResult to filter
            filters : dict of {metadata_key: required_value}

        Returns:
            Filtered list (may be empty if nothing matches).
        """
        if not filters:
            return results

        filtered: List[RetrieverResult] = []
        for r in results:
            # Check all filters against both top-level fields and metadata
            match = True
            for key, value in filters.items():
                # Look in top-level fields first (doc_id, chunk_id, etc.)
                actual = getattr(r, key, None)
                if actual is None:
                    # Fall back to the metadata dict
                    actual = r.metadata.get(key)
                if actual is None:
                    match = False
                    break
                # Case-insensitive substring match
                if str(value).lower() not in str(actual).lower():
                    match = False
                    break
            if match:
                filtered.append(r)

        logger.info(
            "Metadata filter %s: %d -> %d results",
            filters,
            len(results),
            len(filtered),
        )
        return filtered

    # ------------------------------------------------------------------
    #  Context formatting (unchanged)
    # ------------------------------------------------------------------

    def format_context(self, results: List[RetrieverResult], max_chars: int = 3000) -> str:
        """
        Format retrieval results into a context string suitable for LLM
        prompt construction.

        Each chunk is numbered and separated.  The total length is capped
        at max_chars to stay within LLM context limits.

        Args:
            results   : list of RetrieverResult from .query()
            max_chars : maximum total characters in the context block

        Returns:
            Formatted string ready to insert into an LLM prompt.
        """
        parts: List[str] = []
        total = 0
        for i, r in enumerate(results, 1):
            header = f"[Source {i} | score={r.score:.3f} | doc={r.doc_id}]"
            block = f"{header}\n{r.text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)
