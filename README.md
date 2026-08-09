# Multimodal Fake News Detection using RAG

This repository is now organized around one official Phase 4 demo pipeline:

`MMFakeBench sample -> image resolution -> FAISS evidence retrieval -> CLIP multimodal classifier -> prediction + confidence + explanation + evidence`

The project is no longer centered on text-only baselines. The main system uses:
- MMFakeBench JSON records with `text`, `image_path`, `gt_answers`, and `fake_cls`
- a CLIP-based image-text classifier
- optional Google Fact Check Tools evidence for live claim verification
- GDELT live news retrieval for general current-news evidence
- Wikidata trusted knowledge for public-entity status claims
- a local SentenceTransformer + FAISS evidence store
- one shared prediction path for API, CLI, batch inference, and evaluation

## Current Scope

What works now:
- multimodal sample loading from MMFakeBench
- robust image lookup with placeholder fallback when images are missing
- optional live fact-check evidence when `GOOGLE_FACT_CHECK_API_KEY` is configured
- live news and trusted knowledge retrieval for broader claims
- final verdicts are evidence-first; the trained classifier no longer decides the user-facing label when API evidence is missing or inconclusive
- local FAISS evidence retrieval during prediction
- single-sample inference
- batch inference
- evaluation with Accuracy, Precision, Recall, F1, Macro-F1, and confusion matrix
- FastAPI backend and Streamlit demo using the same pipeline

What is intentionally simple:
- the classifier is a lightweight CLIP fusion baseline, not a large fine-tuned VLM
- RAG is text-evidence retrieval that is injected into the text side of the multimodal classifier
- training uses an 80/20 split of `MMFakeBench_val.json` because no official train split is provided here

## Dataset Placement

Place the official files here:
- `dataset/MMFakeBench_val.json`
- `dataset/MMFakeBench_test.json`

For images, either of these layouts is supported:
- `dataset/images/fake/...`, `dataset/images/real/...`, `dataset/images/source/...`
- extracted raw folders such as `dataset/MMFakeBench_val/...` and `dataset/MMFakeBench_test/...`

Recommended setup:
1. Extract `MMFakeBench_val.zip`
2. Extract `MMFakeBench_test.zip`
3. Copy the extracted `fake`, `real`, and `source` folders under `dataset/images/`

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional live fact-check evidence:

```powershell
$env:GOOGLE_FACT_CHECK_API_KEY="your_google_fact_check_api_key"
```

Optional Gemini grounded verifier:

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key"
```

If terminal paste is difficult on Windows, create this local file instead:

```text
secrets/google_fact_check_api_key.txt
```

Put only the API key inside that file. The `secrets/` folder is ignored by git.

For Gemini, use:

```text
secrets/gemini_api_key.txt
```

Useful optional settings:

```powershell
$env:FACT_CHECK_LANGUAGE_CODE="en"
$env:FACT_CHECK_MAX_AGE_DAYS="365"
```

## Official Run Paths

### 1. Build the Evidence Index

Build the FAISS index from the validation split so retrieval is available during inference and demo usage:

```bash
python retrieval/build_index.py --corpus_json dataset/MMFakeBench_val.json
```

You can also build from multiple files if you want a larger local corpus:

```bash
python retrieval/build_index.py --corpus_json dataset/MMFakeBench_val.json dataset/MMFakeBench_test.json
```

Use the validation-only option if you want to avoid test leakage during evaluation.

### 2. Train the Multimodal Classifier

This uses an 80/20 split of `MMFakeBench_val.json` for train/validation:

```bash
python training/train.py --annotation_file dataset/MMFakeBench_val.json --epochs 5 --batch_size 16
```

Outputs:
- best checkpoint: `checkpoints/model_best.pt`
- training metrics: `checkpoints/training_metrics.json`

### 3. Single Inference

```bash
python infer_single.py --text "Jake Davis who has been released from a young offender institution first appeared in court in 2011" --image_path "dataset/images/real/bbc_val_50/BBC_val_0.png"
```

This prints JSON containing:
- predicted label
- confidence
- class probabilities
- explanation
- retrieved evidence
- image status

### 4. Batch Inference

```bash
python infer_batch.py --annotation_file dataset/MMFakeBench_test.json --output_file outputs/batch_predictions.csv
```

Optional demo-friendly limit:

```bash
python infer_batch.py --annotation_file dataset/MMFakeBench_test.json --limit 200
```

### 5. Evaluation

```bash
python evaluation/eval_mm.py --annotation_file dataset/MMFakeBench_test.json
```

Outputs:
- detailed predictions: `outputs/eval_predictions.csv`
- metrics summary: `outputs/eval_metrics.json`

Metrics include:
- Accuracy
- Precision
- Recall
- F1
- Macro-F1
- confusion matrix

Optional faster smoke-test:

```bash
python evaluation/eval_mm.py --annotation_file dataset/MMFakeBench_test.json --limit 200
```

### 6. Demo Backend

```bash
uvicorn backend.main:app --reload --port 8080
```

Health check:

```bash
http://127.0.0.1:8080/health
```

### 7. Streamlit Demo

In a second terminal:

```bash
python -m streamlit run frontend/app.py
```

The demo shows:
- input claim
- optional image
- predicted class
- confidence
- explanation
- live fact-check, live news, trusted knowledge, and/or local FAISS evidence
- image fallback status

## Main Files

Official main path:
- `dataset/mmfakebench.py`
- `models/multimodal_classifier.py`
- `retrieval/build_index.py`
- `retrieval/rag_retriever.py`
- `backend/pipeline.py`
- `backend/main.py`
- `infer_single.py`
- `infer_batch.py`
- `evaluation/eval_mm.py`
- `training/train.py`
- `frontend/app.py`

## Limitations

- Live fact-checking is optional and requires `GOOGLE_FACT_CHECK_API_KEY`; without it, the system falls back to the local FAISS evidence store.
- A missing fact-check or news result is not treated as proof that a claim is true or false; the system can return `Unverified` instead of guessing.
- The CLIP classifier is kept as a secondary project signal, but final demo labels are controlled by external evidence APIs.
- The RAG stage retrieves text evidence only. Image evidence is still handled through the CLIP image encoder rather than a separate image retriever.
- Appending evidence to CLIP text input is a practical baseline, not full multimodal cross-attention RAG fusion.
- Full test-set evaluation can be slow on a laptop because retrieval and multimodal inference run sample by sample through the unified pipeline.
