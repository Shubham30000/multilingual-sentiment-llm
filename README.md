# 🌐 Multilingual Sentiment Analysis — NPPE1 (IIT Madras BS Degree)

> 📊 Focus: Multilingual NLP, low-resource learning, and efficient model adaptation under compute constraints

> 🚀 Achieved **0.95+ Macro F1** using parameter-efficient fine-tuning on a low-resource multilingual dataset — competitive with top leaderboard submissions.

Fine-tuning `google/gemma-3-1b-it` for binary sentiment classification across **13 Indian languages** using QLoRA on Kaggle.

---

## 🏆 Competition Result

| Metric | Score |
|--------|-------|
| Public Leaderboard F1 | **0.9545** |
| Performance | Competitive with top leaderboard submissions |
| Evaluation Metric | Macro F1 Score |

---

## 📌 Problem Statement

This project was part of the **NPPE-1 competition** hosted by the Industry Interaction Cell, IIT Madras BS Degree Program. The task was to build a sentiment classifier for text in 13 Indian languages — classifying each sentence as **Positive (1)** or **Negative (0)**.

The challenge:
- Only **900 labeled training samples** across 13 languages (~69 per language)
- Must use `google/gemma-3-1b-it` as the base model
- All training restricted to **Kaggle Notebooks** (no external GPUs or data)
- Evaluate on **Macro F1 Score**

---

## 🔄 Pipeline Overview

```
Text → Prompt Formatting → Tokenization → QLoRA Fine-Tuning → Inference → Output Parsing → Final Label (0/1)
```

---

## 🗂️ Dataset

| Split | Samples | Columns |
|-------|---------|---------|
| Train | 900 | ID, sentence, label, language |
| Test  | 100 | ID, sentence, language |

**Supported Languages:**

| Code | Language | Code | Language |
|------|----------|------|----------|
| `as` | Assamese | `kn` | Kannada |
| `bd` | Bodo | `ml` | Malayalam |
| `bn` | Bengali | `mr` | Marathi |
| `gu` | Gujarati | `or` | Odia |
| `hi` | Hindi | `pa` | Punjabi |
| `ta` | Tamil | `te` | Telugu |
| `ur` | Urdu | — | — |

---

## 🧠 Approach

### Model
- **Base model:** `google/gemma-3-1b-it` (instruction-tuned, 1B parameters)
- **Fine-tuning method:** QLoRA (4-bit quantization + LoRA adapters)
- **Quantization:** NF4, double quantization, bfloat16 compute

### LoRA Configuration
```
r = 16
lora_alpha = 32
target_modules = [q_proj, k_proj, v_proj, o_proj]
lora_dropout = 0.05
trainable params: ~2.98M / 1.00B total (0.30%)
```

### Prompt Format
Used Gemma's native chat template to align with the model's pretraining objective:
```
<start_of_turn>user
Classify the sentiment of the following Hindi text as exactly one word: Positive or Negative.
Text: <sentence>
<end_of_turn>
<start_of_turn>model
Positive<end_of_turn>
```

### Training
| Hyperparameter | Value |
|----------------|-------|
| Epochs | 4 |
| Batch size | 4 |
| Gradient accumulation | 4 steps |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Optimizer | paged_adamw_8bit |
| Hardware | Tesla P100 16GB (Kaggle) |
| Training time | ~8 minutes |

---

## ⚠️ Challenges & Solutions

- **Multilingual variability:** Model needed to generalize across 13 scripts with minimal data → addressed using instruction-style prompting to standardize inputs regardless of language
- **Inconsistent model outputs** (e.g., `"."`, extra text): Built a post-processing pipeline to enforce strict label extraction from raw generated text
- **Small dataset size (~900 samples):** Used QLoRA to keep trainable parameters under 0.3% of total, preventing overfitting on low-resource data
- **Compute constraints (Kaggle free tier):** Applied 4-bit NF4 quantization + gradient accumulation to fit the full training pipeline within P100 memory limits
- **Dependency conflicts:** Resolved version incompatibilities between `transformers`, `bitsandbytes`, and `torchvision` in the Kaggle environment

---

## 🧠 Why This Approach Works

- **Instruction-tuned base model** — Gemma-3-1B-IT generalizes better across languages when inputs are framed as explicit tasks rather than raw text
- **QLoRA efficiency** — Enables meaningful adaptation without catastrophic forgetting, critical when fine-tuning data is scarce
- **Macro F1 optimization** — Ensures balanced performance across both classes and all 13 languages, not just majority-class accuracy
- **Prompt-based formulation** — Aligns the classification task with the model's pretraining objective, reducing the gap between pretrain and fine-tune distributions

---

## 📊 Results

```
              precision    recall  f1-score

    Negative       0.95      0.94      0.94
    Positive       0.93      0.94      0.94

    accuracy                           0.94
   macro avg       0.94      0.94      0.94
```

---

## 📁 Repository Structure

```
├── notebook.ipynb        # Full Kaggle notebook (EDA + training + inference)
├── README.md             # This file
└── sample_submission.csv # Example submission format
```

---

## ⚙️ Setup & Reproduction

All training was done on **Kaggle Notebooks** using the free GPU tier. To reproduce:

1. Upload `notebook.ipynb` to Kaggle
2. Add the competition dataset
3. Add your HuggingFace token as a Kaggle Secret named `HF_TOKEN`
4. Accept the Gemma model license at [huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
5. Run all cells

**Dependencies:**
```bash
transformers==4.50.0
peft==0.14.0
trl==0.12.0
accelerate==0.34.0
bitsandbytes (latest)
datasets
```

---

## 💡 Key Learnings

- QLoRA is highly effective for low-resource multilingual fine-tuning — less than 0.3% of parameters trained yet strong cross-lingual performance
- Prompt engineering significantly impacts output quality — using the model's native chat format is better than plain text inputs
- Small datasets amplify training variance — always validate locally before using limited competition submissions
- Practical debugging skills matter as much as modeling — environment setup in constrained compute environments is a real challenge

---

## 🔗 Links

- 📦 [Kaggle Competition](https://www.kaggle.com/competitions/nppe-dlp-2026-term-1)
- 🤗 [Gemma 3 1B IT Model](https://huggingface.co/google/gemma-3-1b-it)
- 🏫 [IIT Madras BS Degree Program](https://study.iitm.ac.in/ds/)

---

## 📄 License

This project was developed as part of an academic competition. The dataset is subject to competition rules.
