# MedSynth Dial2Note — SOAP Note Generation

Automatic generation of structured SOAP clinical notes from doctor–patient dialogues, submitted to the **SMM4H 2026 Shared Task**.

---

## How it works

1. **Retrieval** — For each dialogue in the eval set, the most relevant training examples are retrieved using Reciprocal Rank Fusion (RRF) of BM25 and BioLORD dense embeddings (`FremyCompany/BioLORD-2023`). BM25 scoring is vectorised as a single sparse matrix multiplication for efficiency.

2. **Few-shot prompting** — Retrieved examples are injected as alternating `user`/`assistant` turns in the chat prompt, followed by the target dialogue. This outperforms block-style few-shot prompting on chat-tuned models.

3. **Generation** — Prompts are sent to an OpenRouter-hosted LLM (default: `openai/gpt-4o-mini`) with up to 200 concurrent async requests via `aiohttp`.

4. **Evaluation** — Generated notes are scored against reference notes using BLEU, ROUGE-{1,2,L}, METEOR, and a heuristic medical-term F1 (MedCon).

### Experimental grid

| Config | Retrieval | Prompt variant |
|---|---|---|
| `gpt_biolord_k3_turns_eval` | RRF(BM25+BioLORD) k=3 | standard |
| `06a_rrf_biolord_k3_turns` | RRF(BM25+BioLORD) k=3 | standard |
| `06b_rrf_biolord_icd10_boost_k3_turns` | RRF(BM25+BioLORD) k=3 + ICD-10 boost | standard |
| `05c_rrf_bge_k3_turns_medcon_prime` | RRF(BM25+BGE) k=3 | MedCon key-term priming |
| `06d_rrf_biolord_k3_turns_medcon_prime` | RRF(BM25+BioLORD) k=3 | MedCon key-term priming |
| `05f_triple_rrf_k3_turns` | RRF(BM25+BGE+BioLORD) k=3 | standard |
| `05g_rrf_bge_k3_turns_refstyle` | RRF(BM25+BGE) k=3 | reference-style guide |

---

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

---

## Run

```bash
# All experiments, 300 samples
python dial2note.py \
    --train-csv shared_task_train.csv \
    --eval-csv  test_set_participant_version.csv \
    --n-samples 300

# Single experiment
python dial2note.py \
    --experiments gpt_biolord \
    --n-samples 300

# Different model
python dial2note.py \
    --model meta-llama/llama-3.1-8b-instruct \
    --n-samples 300
```

**All flags**

| Flag | Default | Description |
|---|---|---|
| `--train-csv` | `shared_task_train.csv` | Training corpus |
| `--eval-csv` | `test_set_participant_version.csv` | Evaluation / submission corpus |
| `--n-samples` | `300` | Number of eval rows to process |
| `--model` | `openai/gpt-4o-mini` | Any OpenRouter model id |
| `--experiments` | *(all)* | Name prefix(es) to filter experiments |
| `--concurrency` | `200` | Max concurrent API requests |

---

## Outputs

All artifacts are written to `experiment_results/`:

| File | Description |
|---|---|
| `<name>.json` | Config, metrics, and prediction/reference pairs |
| `<name>_submission.csv` | `id, generated_note` — ready for task submission |
| `experiment_summary.csv` | Metric table across all experiments, sorted by Average |
