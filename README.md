# Multimodal Fake News Detection using RAG

This project is a complete, working, end-to-end multimodal fake news detection system using **Text + Image input**, replacing the legacy text-only Logistic Regression baselines. The system uses a modern **Multimodal Baseline** (CLIP image-text encoder with a lightweight MLP classifier head) trained on the **MMFakeBench dataset**, alongside **real RAG functionality** (FAISS + SentenceTransformers) to inject evidential ground-truth into the claim verification pipeline. 

## 🗺️ Fallback Assumptions & Known Limitations
- **No Official Train Split:** The MMFakeBench dataset provided does not possess an explicit `train.json`. We strictly documented this fallback: The dataset parsing automatically utilizes an **80/20 slice** of the validation set (`val.json`) for training and reserving the rest for validation. 
- **Legacy Components:** Any previous Logistic Regression elements have been successfully fully deprecated.

## 📁 Dataset Placement

Ensure your dataset files are loaded as follows inside the `dataset/` directory:
- `dataset/MMFakeBench_val.json`
- `dataset/MMFakeBench_test.json`
- `dataset/images/` (Extracted images from the ZIP files placed here)

## 🛠️ Environment Setup

Create an environment and install tools. Ensure you have the `sentence-transformers` and `faiss-cpu` dependencies for the Real RAG Retreiver:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision transformers pillow pandas scikit-learn faiss-cpu sentence-transformers python-multipart
```

## 🚀 How to Run the System

### 1. Build the Evidence Store (RAG)
To use real RAG retrieval during inference or app usage, build the FAISS index from the dataset texts:
```bash
python retrieval/build_index.py --corpus_json dataset/MMFakeBench_val.json
```

### 2. Training the Multimodal Baseline
Run the training loop to fine-tune the CLIP fusion head using Cross-Entropy Loss:
```bash
python training/train.py --annotation_file dataset/MMFakeBench_val.json --epochs 5 --batch_size 16
```

### 3. Evaluation Script
To calculate Accuracy, Precision, Recall, Macro-F1, and Confusion Matrices on the Test Split:
```bash
python evaluation/eval_mm.py --test_annotation dataset/MMFakeBench_test.json
```

### 4. Inference Scripts

**Single Inference** via CLI (`infer_single.py`):
```bash
python infer_single.py --text "Donald trump is dead" --image_path "dataset/images/example.png"
```
*(This retrieves FAISS evidence automatically and predicts a label!)*

**Batch Inference** via CLI (`infer_batch.py`):
```bash
python infer_batch.py --split test --annotation_file dataset/MMFakeBench_test.json
```
*(Outputs predictions to `outputs/batch_predictions.csv`)*

### 5. Local App / UI Demo

The User Interface of this demo application utilizes the **Stitch MCP Server** generated layouts to interface cleanly with FastAPI.
The UI relies on the local API Backend to process the heavy CLIP inference operations cleanly.

First, **Spin up the Backend API** in one activated terminal on port 8080 (avoids Windows port 8000 hyper-v restrictions):
```bash
uvicorn backend.main:app --reload --port 8080
```

Then, **Run the Streamlit multimodal demo natively** in a separate activated terminal. Bypassing global Anaconda path collisions by triggering via python module logic:
```bash
python -m streamlit run frontend/app.py
```
