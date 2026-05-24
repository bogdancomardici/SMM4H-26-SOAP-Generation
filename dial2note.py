"""
MedSynth Dial2Note — OpenRouter Experiment Pipeline
====================================================

End-to-end pipeline that, for a given doctor–patient dialogue, retrieves the
most relevant training examples, builds a chat-style few-shot prompt, and
queries an OpenRouter-hosted LLM in parallel to produce a structured SOAP
clinical note. 

Inputs
------
- A training CSV with columns ``dialogue``, ``note`` (and optionally an
  ICD-10 code column).
- An evaluation CSV with at least ``dialogue`` (and optionally ``id`` /
  ``note`` for submission building / metric computation respectively).

Outputs (in ``experiment_results/``)
------------------------------------
- ``<experiment>.json``      — config, metrics, and (prediction, reference)
                               pairs for every sample.
- ``<experiment>_submission.csv`` — ``id, generated_note`` file ready for
                               shared-task submission (when ``id`` is
                               present in the eval CSV).
- ``experiment_summary.csv`` — aggregated metric table across experiments.

Reproduce a single run
----------------------
    export OPENROUTER_API_KEY=sk-...
    python dial2note.py \\
        --n-samples 300 \\
        --train-csv shared_task_train.csv \\
        --eval-csv  shared_task_eval.csv \\
        --experiments gpt_biolord
"""

from __future__ import annotations


import argparse
import asyncio
import json
import os
import random
import re
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable


import aiohttp
import evaluate
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

SEED: Final[int] = 42
random.seed(SEED)
np.random.seed(SEED)

# Defaults — every value below can be overridden on the command line.
DEFAULT_MODEL: Final[str] = "openai/gpt-4o-mini"
DEFAULT_TRAIN_CSV: Final[str] = "shared_task_train.csv"
DEFAULT_EVAL_CSV: Final[str] = "shared_task_train.csv"
DEFAULT_N_SAMPLES: Final[int] = 300

OUTPUT_DIR: Final[Path] = Path("experiment_results")
CACHE_DIR: Final[Path] = Path("generation_cache")

# OpenRouter
OPENROUTER_URL: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CONCURRENCY: Final[int] = 200
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_MAX_TOKENS: Final[int] = 3000

# Retrieval
RRF_K: Final[int] = 60
RETRIEVAL_POOL: Final[int] = 50
DEFAULT_ICD10_BOOST: Final[float] = 0.2

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

for _pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DataColumns:
    """Resolved column names for the loaded train/eval frames."""

    dialogue: str = "dialogue"
    note: str = "note"
    icd10: str | None = None


def load_data(
    train_csv: str, eval_csv: str
) -> tuple[pd.DataFrame, pd.DataFrame, DataColumns]:
    """Load the train/eval CSVs and auto-detect the relevant columns."""
    train_df = pd.read_csv(train_csv)[:10]
    eval_df = pd.read_csv(eval_csv)[:10]

    dialogue_col = "dialogue"
    note_col = "note"
    for col in train_df.columns:
        cl = col.strip().lower()
        if cl == "dialogue":
            dialogue_col = col
        if cl == "note":
            note_col = col

    icd10_col: str | None = None
    for col in train_df.columns:
        if "icd" in col.lower():
            icd10_col = col
            break

    train_df = train_df.dropna(subset=[dialogue_col])
    eval_df = eval_df.dropna(subset=[dialogue_col])

    cols = DataColumns(dialogue=dialogue_col, note=note_col, icd10=icd10_col)
    print(f"Loaded train: {len(train_df)} | eval: {len(eval_df)}")
    print(
        f"Columns: dialogue='{cols.dialogue}', note='{cols.note}', icd10='{cols.icd10}'"
    )
    return train_df, eval_df, cols


# ════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ════════════════════════════════════════════════════════════════════════════

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

# Heuristic medical-term regexes used by the MedCon proxy metric (and by the
# optional pre-processing key-term priming prompt).
MEDICAL_TERM_PATTERNS: Final[list[str]] = [
    r"\b\w+(?:ol|in|ine|ide|ate|one|pam|lol|pril|artan|statin|mycin|cillin|oxacin|azole|prazole)\b",
    r"\b\d+\s*(?:mg|mcg|ml|mL|units?|tablets?|capsules?|puffs?)\b",
    r"\b(?:blood pressure|BP|heart rate|HR|respiratory rate|RR|temperature|SpO2|pulse ox|BMI)\b",
    r"\b(?:hypertension|diabetes|asthma|COPD|depression|anxiety|arthritis|pneumonia|infection)\b",
    r"\b(?:CBC|BMP|CMP|TSH|A1c|HbA1c|WBC|RBC|hemoglobin|creatinine|glucose|cholesterol)\b",
    r"\b(?:X-ray|MRI|CT scan|ultrasound|ECG|EKG|spirometry|colonoscopy|biopsy)\b",
    r"\b(?:ibuprofen|acetaminophen|metformin|lisinopril|amlodipine|omeprazole|atorvastatin|losartan|levothyroxine|metoprolol)\b",
    r"\b(?:prednisone|amoxicillin|azithromycin|gabapentin|hydrochlorothiazide|sertraline|fluoxetine|duloxetine)\b",
    r"\b(?:anemia|hypothyroidism|hyperthyroidism|GERD|reflux|migraine|seizure|stroke|embolism|thrombosis)\b",
    r"\b(?:edema|effusion|stenosis|fracture|dislocation|sprain|contusion|laceration|abscess)\b",
    r"\b(?:urinalysis|BUN|GFR|troponin|D-dimer|INR|PT|PTT|ESR|CRP|procalcitonin)\b",
    r"\b(?:mammogram|colonoscopy|endoscopy|bronchoscopy|arthroscopy|laparoscopy|echocardiogram)\b",
    r"\b\d+/\d+\s*(?:mmHg)?\b",
    r"\b\d+\.?\d*\s*(?:mg/dL|mmol/L|mEq/L|g/dL|%)\b",
]


def extract_medical_terms(text: str) -> set[str]:
    """Return the set of unique medical-term mentions found in ``text``."""
    terms: set[str] = set()
    for pattern in MEDICAL_TERM_PATTERNS:
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        terms.update(m.lower().strip() for m in matches)
    return terms


def compute_medcon(predictions: list[str], references: list[str]) -> float:
    """Token-level F1 between extracted medical terms in predictions vs. refs."""
    f1_scores: list[float] = []
    for pred, ref in zip(predictions, references):
        pred_terms = extract_medical_terms(pred)
        ref_terms = extract_medical_terms(ref)
        if not pred_terms and not ref_terms:
            f1_scores.append(1.0)
        elif not pred_terms or not ref_terms:
            f1_scores.append(0.0)
        else:
            true_positives = len(pred_terms & ref_terms)
            precision = true_positives / len(pred_terms)
            recall = true_positives / len(ref_terms)
            denom = precision + recall
            f1_scores.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return float(np.mean(f1_scores))


def evaluate_predictions(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Compute BLEU, ROUGE-{1,2,L}, METEOR, MedCon and their average."""
    bleu = bleu_metric.compute(predictions=predictions, references=references)
    rouge = rouge_metric.compute(predictions=predictions, references=references)
    meteor = meteor_metric.compute(predictions=predictions, references=references)
    results: dict[str, float] = {
        "BLEU": bleu["score"] / 100.0,
        "ROUGE-1": rouge["rouge1"],
        "ROUGE-2": rouge["rouge2"],
        "ROUGE-L": rouge["rougeL"],
        "METEOR": meteor["meteor"],
        "MedCon": compute_medcon(predictions, references),
    }
    results["Average"] = float(np.mean(list(results.values())))
    return results


# ════════════════════════════════════════════════════════════════════════════
# Retrieval index
# ════════════════════════════════════════════════════════════════════════════


class RetrievalIndex:
    """Hybrid retrieval index supporting BM25, BGE, BioLORD, and RRF fusions.

    Notable engineering detail (worth highlighting in the paper): BM25
    scoring across many queries is reformulated as a single sparse
    matrix multiplication.  We pre-compute a ``(n_terms × n_docs)`` matrix
    of Okapi document weights once; per-query scoring then reduces to
    constructing a sparse query-term/IDF matrix and one matmul, yielding
    orders-of-magnitude speedups over per-query token-loop scoring.
    """

    BGE_MODEL: Final[str] = "BAAI/bge-base-en-v1.5"
    BIOLORD_MODEL: Final[str] = "FremyCompany/BioLORD-2023"

    def __init__(
        self,
        train_df: pd.DataFrame,
        dialogue_col: str,
        note_col: str,
        icd10_col: str | None = None,
    ) -> None:
        self.train_df = train_df.reset_index(drop=True)
        self.dialogue_col = dialogue_col
        self.note_col = note_col
        self.icd10_col = icd10_col
        # Coerce dialogues to str (replace NaN with empty string).
        self.dialogues: list[str] = [
            d if isinstance(d, str) else ("" if pd.isna(d) else str(d))
            for d in self.train_df[dialogue_col].tolist()
        ]
        self.notes = self.train_df[note_col].tolist()
        self.n_docs = len(self.dialogues)

        # Lazily-initialised caches.
        self._bm25: Any = None
        self._bm25_tokenize: Callable[[str], list[str]] | None = None
        self._bm25_matrix: Any = None  # sparse (n_terms x n_docs)
        self._vocab: dict[str, int] | None = None
        self._idf_vec: np.ndarray | None = None
        self._dense_embs: dict[str, np.ndarray] = {}
        self._dense_models: dict[str, Any] = {}
        self._icd10_index: dict[Any, list[int]] | None = None
        self._icd10_cat_index: dict[str, list[int]] | None = None

    # ── BM25 ──

    def _get_bm25(self) -> Any:
        """Build (once) and return the BM25Okapi index + sparse weight matrix."""
        if self._bm25 is not None:
            return self._bm25

        from rank_bm25 import BM25Okapi
        from nltk.tokenize import word_tokenize
        import scipy.sparse as sp

        print("  Building BM25 index...")
        tokenized = [
            word_tokenize(d.lower())
            for d in tqdm(self.dialogues, desc="  BM25 tokenize")
            if isinstance(d, str)
        ]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_tokenize = lambda t: word_tokenize(t.lower())

        bm25 = self._bm25
        vocab: dict[str, int] = {}
        for doc_freqs in bm25.doc_freqs:
            for word in doc_freqs:
                if word not in vocab:
                    vocab[word] = len(vocab)
        n_terms = len(vocab)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for doc_idx, (doc_freqs, doc_len) in enumerate(
            zip(bm25.doc_freqs, bm25.doc_len)
        ):
            dl_factor = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
            for word, tf in doc_freqs.items():
                rows.append(vocab[word])
                cols.append(doc_idx)
                data.append((tf * (bm25.k1 + 1)) / (tf + dl_factor))

        self._bm25_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_terms, self.n_docs), dtype=np.float32
        )
        self._vocab = vocab
        self._idf_vec = np.zeros(n_terms, dtype=np.float32)
        for word, idx in vocab.items():
            if word in bm25.idf:
                self._idf_vec[idx] = bm25.idf[word]

        return self._bm25

    def _bm25_batch_scores(self, queries: list[str]) -> np.ndarray:
        """Vectorised BM25 scoring: one sparse matmul for the whole batch."""
        import scipy.sparse as sp

        assert self._vocab is not None and self._idf_vec is not None
        assert self._bm25_tokenize is not None

        n_q = len(queries)
        n_terms = len(self._vocab)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for q_idx, query in enumerate(queries):
            for token in set(self._bm25_tokenize(query)):
                if token in self._vocab:
                    t_idx = self._vocab[token]
                    rows.append(q_idx)
                    cols.append(t_idx)
                    data.append(self._idf_vec[t_idx])
        query_mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_q, n_terms), dtype=np.float32
        )
        return (query_mat @ self._bm25_matrix).toarray()  # (n_queries, n_docs)

    # ── Dense ──

    def _get_dense(self, model_name: str) -> tuple[Any, np.ndarray]:
        """Lazily encode the corpus with ``model_name`` and return (model, embs)."""
        if model_name not in self._dense_embs:
            from sentence_transformers import SentenceTransformer

            if model_name not in self._dense_models:
                print(f"  Loading: {model_name}")
                self._dense_models[model_name] = SentenceTransformer(model_name)
            model = self._dense_models[model_name]
            print(f"  Encoding {self.n_docs} dialogues with {model_name}...")
            self._dense_embs[model_name] = model.encode(
                self.dialogues,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        return self._dense_models[model_name], self._dense_embs[model_name]

    # ── ICD-10 ──

    def _get_icd10_index(self) -> dict[Any, list[int]]:
        """Map ICD-10 code → list of training-row indices with that code."""
        if self._icd10_index is None and self.icd10_col:
            self._icd10_index = (
                self.train_df.groupby(self.icd10_col)
                .apply(lambda g: g.index.tolist())
                .to_dict()
            )
        return self._icd10_index or {}

    # ── Helpers ──

    def _format(
        self, indices: Iterable[int], scores: np.ndarray | None = None
    ) -> list[dict[str, Any]]:
        """Render a list of training-row indices as retrieval result dicts."""
        return [
            {
                "dialogue": self.train_df.iloc[idx][self.dialogue_col],
                "note": self.train_df.iloc[idx][self.note_col],
                "icd10": self.train_df.iloc[idx].get(self.icd10_col, "")
                if self.icd10_col
                else "",
                "score": float(scores[i]) if scores is not None else 1.0,
            }
            for i, idx in enumerate(indices)
        ]

    def _rrf_from_ranked_lists(
        self, ranked_lists: list[list[int]], rrf_k: int = RRF_K
    ) -> np.ndarray:
        """Reciprocal Rank Fusion of ``ranked_lists`` → score vector per doc."""
        scores = np.zeros(self.n_docs)
        for ranked in ranked_lists:
            for rank, doc_idx in enumerate(ranked):
                scores[doc_idx] += 1.0 / (rrf_k + rank + 1)
        return scores

    def _apply_icd10_boost(
        self, rrf: np.ndarray, query_icd10: str, boost: float
    ) -> None:
        """Add an additive ICD-10 boost in-place to the RRF score vector.

        Exact-code matches receive the full ``boost``; same-category (first
        three chars) matches receive 30%.  At ``rrf_k=60`` the highest
        single-list contribution to RRF is ``1/61 ≈ 0.016``, so the default
        ``boost=0.2`` is large enough to materially promote ICD-10 matches
        while still respecting textual relevance.
        """
        if not (self.icd10_col and query_icd10):
            return
        icd10_idx = self._get_icd10_index()
        for idx in icd10_idx.get(query_icd10, []):
            rrf[idx] += boost
        prefix = str(query_icd10)[:3]
        for code, idxs in icd10_idx.items():
            if str(code)[:3] == prefix and code != query_icd10:
                for idx in idxs:
                    rrf[idx] += boost * 0.3

    # ── Batch retrieval ──

    def retrieve_batch(
        self,
        queries: list[str],
        method: str,
        k: int = 3,
        pool: int = RETRIEVAL_POOL,
        query_icd10s: list[str] | None = None,
        boost: float = DEFAULT_ICD10_BOOST,
    ) -> list[list[dict[str, Any]]]:
        """Batch retrieval: encode once, score once, fuse per query."""
        from sklearn.metrics.pairwise import cosine_similarity

        n = len(queries)
        self._get_bm25()  # ensure BM25 + sparse matrix are built

        bio_all_sims: np.ndarray | None = None
        bge_all_sims: np.ndarray | None = None
        if method in ("rrf_biolord", "rrf_biolord_icd10", "triple_rrf"):
            model, bio_embs = self._get_dense(self.BIOLORD_MODEL)
            print("  BioLORD encode queries...")
            q_bio = model.encode(
                queries, batch_size=64, show_progress_bar=True, convert_to_numpy=True
            )
            bio_all_sims = cosine_similarity(q_bio, bio_embs)
        if method in ("rrf_bge", "triple_rrf"):
            model, bge_embs = self._get_dense(self.BGE_MODEL)
            print("  BGE encode queries...")
            q_bge = model.encode(
                queries, batch_size=64, show_progress_bar=True, convert_to_numpy=True
            )
            bge_all_sims = cosine_similarity(q_bge, bge_embs)

        print("  BM25 batch scoring...")
        bm25_all_scores = self._bm25_batch_scores(queries)

        results: list[list[dict[str, Any]]] = []
        for i in tqdm(range(n), desc="  RRF fusion"):
            bm25_ranked = np.argsort(bm25_all_scores[i])[::-1][:pool].tolist()

            if method == "rrf_bge":
                assert bge_all_sims is not None
                dense_ranked = np.argsort(bge_all_sims[i])[::-1][:pool].tolist()
                rrf = self._rrf_from_ranked_lists([dense_ranked, bm25_ranked])
            elif method == "rrf_biolord":
                assert bio_all_sims is not None
                dense_ranked = np.argsort(bio_all_sims[i])[::-1][:pool].tolist()
                rrf = self._rrf_from_ranked_lists([dense_ranked, bm25_ranked])
            elif method == "rrf_biolord_icd10":
                assert bio_all_sims is not None
                dense_ranked = np.argsort(bio_all_sims[i])[::-1][:pool].tolist()
                rrf = self._rrf_from_ranked_lists([dense_ranked, bm25_ranked])
                query_icd10 = query_icd10s[i] if query_icd10s else ""
                self._apply_icd10_boost(rrf, query_icd10, boost)
            elif method == "triple_rrf":
                assert bio_all_sims is not None and bge_all_sims is not None
                bio_ranked = np.argsort(bio_all_sims[i])[::-1][:pool].tolist()
                bge_ranked = np.argsort(bge_all_sims[i])[::-1][:pool].tolist()
                rrf = self._rrf_from_ranked_lists([bge_ranked, bio_ranked, bm25_ranked])
            else:
                raise ValueError(f"Unknown retrieval: {method}")

            top_k = np.argsort(rrf)[::-1][:k]
            results.append(self._format(top_k, rrf[top_k]))

        return results


# ════════════════════════════════════════════════════════════════════════════
# Prompt templates
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT: Final[str] = (
    "You are a professional medical transcriptionist and clinical documentation specialist. "
    "Your task is to transcribe and organize information from recorded doctor-patient conversations into "
    "structured SOAP notes. This is a documentation task — you are NOT providing medical advice.\n\n"
    "CRITICAL RULES:\n"
    "- Transcribe ONLY information explicitly stated in the dialogue. Do NOT infer or hallucinate.\n"
    '- If information is not in the dialogue, write "Not documented" — never invent it.\n'
    "- You MUST always complete the full note with ALL sections. Never refuse or decline.\n"
    "- Start directly with the note content. No preamble or closing remarks."
)


def prompt_full_note(dialogue: str) -> str:
    """Baseline SOAP-note generation prompt for a single dialogue."""
    return f"""Generate a complete SOAP clinical note from this doctor-patient conversation.

The note MUST contain exactly these four sections: Subjective, Objective, Assessment, Plan.

Guidelines:
- Include ONLY information from the dialogue. Do NOT hallucinate.
- Use professional medical terminology.
- Place information in the correct SOAP section.
- Be thorough — capture all clinically relevant details.

--- DIALOGUE ---
{dialogue}
--- END DIALOGUE ---

Generate the SOAP note now:"""


def prompt_full_note_with_medcon_priming(dialogue: str, key_terms: list[str]) -> str:
    """SOAP-note prompt that pre-injects extracted medical terms as a checklist.

    A post-hoc LLM rewrite to enforce medical-term coverage causes style
    drift.  Instead, we *tell* the model which medical terms it should
    ensure appear in the note — guiding attention without a second
    generation pass.
    """
    terms_str = ", ".join(key_terms[:20])  # cap to avoid prompt bloat
    return f"""Generate a complete SOAP clinical note from this doctor-patient conversation.

The note MUST contain exactly these four sections: Subjective, Objective, Assessment, Plan.

KEY MEDICAL TERMS from the dialogue (ensure these appear in the appropriate sections):
{terms_str}

Guidelines:
- Include ONLY information from the dialogue. Do NOT hallucinate.
- Use professional medical terminology.
- Place information in the correct SOAP section.
- Be thorough — capture all clinically relevant details.
- Make sure the key medical terms listed above are reflected in the note where clinically appropriate.

--- DIALOGUE ---
{dialogue}
--- END DIALOGUE ---

Generate the SOAP note now:"""


def prompt_reference_style_matched(dialogue: str, reference_style_guide: str) -> str:
    """SOAP-note prompt that embeds a style guide derived from training notes.

    The style guide describes the exact formatting conventions (header
    style, sub-section structure, list formatting) found in the reference
    notes, so the model mimics them for better lexical overlap.
    """
    return f"""Generate a complete SOAP clinical note from this doctor-patient conversation.

The note MUST follow this exact formatting style:
{reference_style_guide}

Guidelines:
- Include ONLY information from the dialogue. Do NOT hallucinate.
- Use professional medical terminology.
- Place information in the correct SOAP section.
- Be thorough — capture all clinically relevant details.
- Match the formatting conventions described above as closely as possible.

--- DIALOGUE ---
{dialogue}
--- END DIALOGUE ---

Generate the SOAP note now:"""


def build_fewshot_turns(
    dialogue: str,
    examples: list[dict[str, Any]],
    prompt_fn: Callable[[str], str] = prompt_full_note,
) -> list[dict[str, str]]:
    """Few-shot examples rendered as alternating user/assistant chat turns.

    Chat-tuned models treat these as genuine demonstrations rather than a
    wall of text, which is why this is the default few-shot style for the
    active experiments.
    """
    messages: list[dict[str, str]] = []
    for ex in examples:
        messages.append({"role": "user", "content": prompt_fn(ex["dialogue"])})
        messages.append({"role": "assistant", "content": ex["note"]})
    return messages


def extract_key_terms_from_dialogue(dialogue: str) -> list[str]:
    """Extract the sorted, unique medical terms in ``dialogue`` for priming."""
    return sorted(extract_medical_terms(dialogue))


# ── Reference-note style analysis (consumed by prompt_reference_style_matched)


def analyze_reference_style(
    train_df: pd.DataFrame, note_col: str, n_sample: int = 200
) -> str:
    """Analyse formatting patterns in training notes; return a style guide string."""
    sample = train_df.sample(min(n_sample, len(train_df)), random_state=SEED)
    notes = sample[note_col].tolist()

    bold_numbered = 0  # **1. Subjective:**
    bold_only = 0  # **Subjective:**
    hash_headers = 0  # ## Subjective
    plain_colon = 0  # Subjective:

    has_cc = 0
    has_hpi = 0
    has_ros = 0
    has_vitals_subsection = 0
    uses_bullets = 0
    uses_dashes = 0

    for note in notes:
        if re.search(r"\*\*\d+\.\s*Subjective", note):
            bold_numbered += 1
        elif re.search(r"\*\*\s*Subjective", note):
            bold_only += 1
        elif re.search(r"#{1,3}\s*Subjective", note):
            hash_headers += 1
        elif re.search(r"Subjective\s*:", note):
            plain_colon += 1

        if re.search(r"Chief\s+Complaint|CC[:\)]", note, re.IGNORECASE):
            has_cc += 1
        if re.search(r"History\s+of\s+Present\s+Illness|HPI", note, re.IGNORECASE):
            has_hpi += 1
        if re.search(r"Review\s+of\s+Systems|ROS", note, re.IGNORECASE):
            has_ros += 1
        if re.search(r"Vital\s+Signs", note, re.IGNORECASE):
            has_vitals_subsection += 1
        if re.search(r"^\s*[-•]", note, re.MULTILINE):
            uses_bullets += 1
        if re.search(r"^\s*-\s", note, re.MULTILINE):
            uses_dashes += 1

    n = len(notes)
    header_counts = {
        "**N. Section:**": bold_numbered,
        "**Section:**": bold_only,
        "## Section": hash_headers,
        "Section:": plain_colon,
    }
    dominant_header = max(header_counts, key=header_counts.get)

    parts = [
        f"SECTION HEADERS: Use the format {dominant_header} for each SOAP section.",
        "  Example: **1. Subjective:**",
    ]
    if has_cc / n > 0.5:
        parts.append(
            "SUBJECTIVE SUB-SECTIONS: Include **Chief Complaint (CC):** and "
            "**History of Present Illness (HPI):** as sub-headers."
        )
    if has_ros / n > 0.4:
        parts.append(
            "Include **Review of Systems (ROS):** as a sub-section under Subjective."
        )
    if has_vitals_subsection / n > 0.4:
        parts.append("OBJECTIVE: Include **Vital Signs:** as a sub-section.")
    if uses_dashes / n > 0.5:
        parts.append(
            "LISTS: Use dash-prefixed lists (- item) for medications, findings, and plans."
        )
    elif uses_bullets / n > 0.5:
        parts.append("LISTS: Use bullet points for medications, findings, and plans.")

    style_guide = "\n".join(parts)

    print(f"\n  Reference style analysis ({n} notes):")
    print(f"    Header style: {dominant_header} ({header_counts[dominant_header]}/{n})")
    print(f"    Has CC: {has_cc}/{n}, HPI: {has_hpi}/{n}, ROS: {has_ros}/{n}")
    print(f"    Vitals sub-section: {has_vitals_subsection}/{n}")
    print(f"    Uses dashes: {uses_dashes}/{n}")

    return style_guide


# ════════════════════════════════════════════════════════════════════════════
# Post-processing
# ════════════════════════════════════════════════════════════════════════════


def clean_note(text: str) -> str:
    """Strip common LLM preamble / disclaimer phrasings from a generated note."""
    preamble_patterns = [
        r"^(?:Here is|Below is|The following is|I\'ve generated|Based on).*?(?=\*?\*?\d*\.?\s*(?:Subjective|1\.))",
        r"^(?:SOAP Note|Clinical Note|Medical Note)\s*:?\s*\n*",
    ]
    trailing_patterns = [
        r"\n*(?:Note:|Please note:|Disclaimer:).*$",
        r"\n*(?:This (?:note|plan) was (?:discussed|reviewed)).*$",
        r"\n*---\s*$",
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    for pat in trailing_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════
# OpenRouter clients
# ════════════════════════════════════════════════════════════════════════════


def _require_api_key() -> str:
    """Read the OpenRouter API key from the environment, or raise."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Export it before running this script."
        )
    return api_key


def _build_openrouter_payload(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Shared request shape for the sync and async OpenRouter helpers."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return headers, payload


async def openrouter_generate_async(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
) -> str:
    """Asynchronous OpenRouter chat-completion call (single message)."""
    headers, payload = _build_openrouter_payload(
        messages, model, temperature, max_tokens, api_key or _require_api_key()
    )
    async with session.post(OPENROUTER_URL, headers=headers, json=payload) as response:
        response.raise_for_status()
        result = await response.json()
        return result["choices"][0]["message"]["content"].strip()


async def generate_all_async(
    messages_list: list[list[dict[str, str]]],
    sample_ids: list[Any] | None,
    *,
    model: str,
    temperature: float,
    api_key: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[list[str], list[Any]]:
    """Run ``openrouter_generate_async`` over many prompts with bounded concurrency.

    Returns ``(responses, failed_ids)``.  Failed samples produce an empty
    string in ``responses`` at the original index.
    """
    semaphore = asyncio.Semaphore(concurrency)
    failed_ids: list[Any] = []

    async with aiohttp.ClientSession() as session:

        async def _bounded(msgs: list[dict[str, str]], idx: int) -> str:
            async with semaphore:
                try:
                    return await openrouter_generate_async(
                        session,
                        msgs,
                        temperature=temperature,
                        api_key=api_key,
                        model=model,
                    )
                except Exception as exc:
                    sid = sample_ids[idx] if sample_ids else idx
                    print(f"  WARNING: generation failed for id={sid}: {exc}")
                    failed_ids.append(sid)
                    return ""

        tasks = [_bounded(msgs, i) for i, msgs in enumerate(messages_list)]
        results = await asyncio.gather(*tasks)
    return results, failed_ids


# ════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentConfig:
    """Single row of the experimental grid."""

    name: str
    description: str
    retrieval_method: str  # rrf_bge | rrf_biolord | rrf_biolord_icd10 | triple_rrf
    retrieval_k: int = 3
    use_medcon_priming: bool = False
    use_reference_style: bool = False
    temperature: float = DEFAULT_TEMPERATURE


def _build_messages_for_sample(
    dialogue: str,
    examples: list[dict[str, Any]],
    config: ExperimentConfig,
    reference_style_guide: str,
) -> list[dict[str, str]]:
    """Assemble the system + few-shot + user prompt chat-message list."""
    if config.use_medcon_priming:
        key_terms = extract_key_terms_from_dialogue(dialogue)
        user_prompt = prompt_full_note_with_medcon_priming(dialogue, key_terms)
    elif config.use_reference_style:
        user_prompt = prompt_reference_style_matched(dialogue, reference_style_guide)
    else:
        user_prompt = prompt_full_note(dialogue)

    fewshot_msgs = build_fewshot_turns(dialogue, examples, prompt_fn=prompt_full_note)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(fewshot_msgs)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def run_experiment(
    config: ExperimentConfig,
    eval_df: pd.DataFrame,
    retrieval_index: RetrievalIndex,
    cols: DataColumns,
    *,
    model: str,
    api_key: str,
    reference_style_guide: str = "",
    n_samples: int = DEFAULT_N_SAMPLES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, float]:
    """Run one ``ExperimentConfig`` end-to-end and persist its artifacts."""
    sample = eval_df.head(n_samples)
    has_references = cols.note in sample.columns
    references: list[str] = sample[cols.note].tolist() if has_references else []
    ids = sample["id"].tolist() if "id" in sample.columns else None

    print(f"\n{'=' * 70}\n  {config.name}\n  {config.description}\n{'=' * 70}")

    # ── Phase 1: batch retrieval + prompt assembly ──
    dialogues: list[str] = sample[cols.dialogue].tolist()
    query_icd10s: list[str] = (
        sample[cols.icd10].fillna("").tolist() if cols.icd10 else [""] * len(sample)
    )

    print(f"  Batch retrieval for {len(dialogues)} samples...")
    all_examples = retrieval_index.retrieve_batch(
        dialogues,
        method=config.retrieval_method,
        k=config.retrieval_k,
        query_icd10s=query_icd10s,
    )
    all_messages = [
        _build_messages_for_sample(dialogue, examples, config, reference_style_guide)
        for dialogue, examples in zip(dialogues, all_examples)
    ]

    # ── Phase 2: parallel generation via aiohttp ──
    print(f"  Generating {len(all_messages)} predictions in parallel...")
    raw_preds, failed_ids = asyncio.run(
        generate_all_async(
            all_messages,
            ids,
            model=model,
            temperature=config.temperature,
            api_key=api_key,
            concurrency=concurrency,
        )
    )

    if failed_ids:
        failed_path = OUTPUT_DIR / f"{config.name}_failed_ids.json"
        with open(failed_path, "w") as f:
            json.dump({"failed_ids": failed_ids, "count": len(failed_ids)}, f, indent=2)
        print(
            f"  WARNING: {len(failed_ids)} predictions failed. Saved to {failed_path}"
        )

    # ── Phase 3: post-process, evaluate, persist ──
    predictions = [clean_note(p) for p in raw_preds]

    results: dict[str, float] = {}
    if has_references:
        results = evaluate_predictions(predictions, references)
        print(f"\n  Results for {config.name}:")
        for k, v in results.items():
            print(f"    {k}: {v:.4f}")
    else:
        print("\n  No reference notes available — skipping evaluation.")

    save_path = OUTPUT_DIR / f"{config.name}.json"
    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "metrics": {k: float(v) for k, v in results.items()},
                "predictions": [
                    {"prediction": p, "reference": r}
                    for p, r in zip(predictions, references)
                ]
                if has_references
                else [{"prediction": p} for p in predictions],
            },
            f,
            indent=2,
        )
    print(f"  Saved to {save_path}")

    if "id" in sample.columns:
        submission_df = pd.DataFrame(
            {"id": sample["id"].tolist(), "generated_note": predictions}
        )
        submission_path = OUTPUT_DIR / f"{config.name}_submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"  Submission saved to {submission_path}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# Experiment definitions
# ════════════════════════════════════════════════════════════════════════════


def get_experiments() -> list[ExperimentConfig]:
    """Active experimental grid for this run.

    Archived configurations remain in the source below so the full paper
    grid is recoverable from a single file.
    """
    return [
        # Headline configuration: BioLORD-RRF retrieval with k=3 turns few-shot.
        ExperimentConfig(
            name="gpt_biolord_k3_turns_eval",
            description="RRF(BM25+BioLORD) k=3, few-shot as turns",
            retrieval_method="rrf_biolord",
            retrieval_k=3,
        ),
        # Best retrieval signal (BioLORD) + best prompt style (turns).
        ExperimentConfig(
            name="06a_rrf_biolord_k3_turns",
            description="RRF(BM25+BioLORD) k=3, few-shot as turns",
            retrieval_method="rrf_biolord",
            retrieval_k=3,
        ),
        # Same as above with a strong ICD-10 boost.
        ExperimentConfig(
            name="06b_rrf_biolord_icd10_boost_k3_turns",
            description="RRF(BM25+BioLORD) k=3, ICD-10 boost=0.2, turns",
            retrieval_method="rrf_biolord_icd10",
            retrieval_k=3,
        ),
        # MedCon pre-processing (key terms injected) on BGE retrieval.
        ExperimentConfig(
            name="05c_rrf_bge_k3_turns_medcon_prime",
            description="RRF(BM25+BGE) k=3, turns + MedCon key terms pre-injected",
            retrieval_method="rrf_bge",
            retrieval_k=3,
            use_medcon_priming=True,
        ),
        # 05d: MedCon pre-processing on BioLORD retrieval.
        ExperimentConfig(
            name="06d_rrf_biolord_k3_turns_medcon_prime",
            description="RRF(BM25+BioLORD) k=3, turns + MedCon key terms pre-injected",
            retrieval_method="rrf_biolord",
            retrieval_k=3,
            use_medcon_priming=True,
        ),
        # 05f: Triple RRF (BM25 + BGE + BioLORD) — maximum retrieval signal.
        ExperimentConfig(
            name="05f_triple_rrf_k3_turns",
            description="Triple-RRF(BM25+BGE+BioLORD) k=3, turns",
            retrieval_method="triple_rrf",
            retrieval_k=3,
        ),
        # 05g: Reference-style-matched prompt with BGE-RRF.
        ExperimentConfig(
            name="05g_rrf_bge_k3_turns_refstyle",
            description="RRF(BM25+BGE) k=3, turns + reference-style-matched prompt",
            retrieval_method="rrf_bge",
            retrieval_k=3,
            use_reference_style=True,
        ),
    ]


# ════════════════════════════════════════════════════════════════════════════
# CLI / main
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class CliArgs:
    """Typed view over ``argparse``'s namespace."""

    n_samples: int
    model: str
    train_csv: str
    eval_csv: str
    experiments: list[str] | None
    concurrency: int


def parse_args() -> CliArgs:
    """Parse command-line arguments and return them as a typed dataclass."""
    parser = argparse.ArgumentParser(
        description="MedSynth Dial2Note — OpenRouter pipeline"
    )
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter model id (e.g. openai/gpt-4o-mini)",
    )
    parser.add_argument("--train-csv", type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval-csv", type=str, default=DEFAULT_EVAL_CSV)
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        help="Run only experiments whose name starts with any of these prefixes.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum concurrent OpenRouter requests.",
    )
    args = parser.parse_args()
    return CliArgs(
        n_samples=args.n_samples,
        model=args.model,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        experiments=args.experiments,
        concurrency=args.concurrency,
    )


def print_summary(all_results: dict[str, dict[str, float]]) -> None:
    """Pretty-print the cross-experiment metric table and write it to CSV."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT SUMMARY")
    print("=" * 80)

    summary_df = pd.DataFrame(all_results).T.sort_values("Average", ascending=False)
    print(summary_df.round(4).to_string())

    summary_path = OUTPUT_DIR / "experiment_summary.csv"
    summary_df.round(4).to_csv(summary_path)
    print(f"\nSaved to {summary_path}")

    best_name = summary_df.index[0]
    best_avg = float(summary_df.iloc[0]["Average"])
    print(f"\nBest: {best_avg:.4f} ({best_name})")


def main() -> None:
    args = parse_args()
    api_key = _require_api_key()

    train_df, eval_df, cols = load_data(args.train_csv, args.eval_csv)

    print("\nAnalyzing reference note formatting...")
    reference_style_guide = analyze_reference_style(train_df, cols.note)

    print("\nBuilding retrieval index...")
    ridx = RetrievalIndex(train_df, cols.dialogue, cols.note, cols.icd10)

    experiments = get_experiments()
    if args.experiments:
        experiments = [
            e
            for e in experiments
            if any(e.name.startswith(p) for p in args.experiments)
        ]

    print(
        f"\nRunning {len(experiments)} experiments, "
        f"{args.n_samples} samples each, model='{args.model}'...\n"
    )

    all_results: dict[str, dict[str, float]] = {}

    for config in experiments:
        try:
            t0 = time.time()
            results = run_experiment(
                config,
                eval_df,
                ridx,
                cols,
                model=args.model,
                api_key=api_key,
                reference_style_guide=reference_style_guide,
                n_samples=args.n_samples,
                concurrency=args.concurrency,
            )
            results["time_seconds"] = time.time() - t0
            all_results[config.name] = results
        except Exception as exc:
            print(f"  ERROR in {config.name}: {exc}")
            traceback.print_exc()

    print_summary(all_results)


if __name__ == "__main__":
    main()
