# Project 1: Fine-tuning Mistral 7B with QLoRA 🚀

A complete end-to-end project for fine-tuning **Mistral-7B-Instruct-v0.2** using **QLoRA** (Quantized Low-Rank Adaptation) on a custom **Hinglish Q&A dataset** of 500 entries covering Yoga, Vedanta, Tantra, and Indian Philosophy.

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Dataset](#-dataset)
- [Training Configuration](#-training-configuration)
- [Setup & Installation](#-setup--installation)
- [Training on Google Colab](#-training-on-google-colab)
- [Inference & Testing](#-inference--testing)
- [Saved Model on Hugging Face](#-saved-model-on-hugging-face)
- [Errors Encountered & Solutions](#-errors-encountered--solutions)
- [Results](#-results)
- [Next Steps](#-next-steps)

---

## 🎯 Overview

| Item | Details |
|---|---|
| **Base Model** | `mistralai/Mistral-7B-Instruct-v0.2` |
| **Fine-tuning Method** | QLoRA (4-bit quantization + LoRA adapters) |
| **Dataset** | 500 custom Hinglish Q&A pairs (`dataset.jsonl`) |
| **Training Hardware** | Google Colab T4 GPU (15GB VRAM) |
| **Trained Adapter** | [sushant2/mistral-hinglish-finetuned](https://huggingface.co/sushant2/mistral-hinglish-finetuned) |
| **Training Time** | ~26 minutes (3 epochs) / ~60 minutes (7 epochs) |

### What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) is a memory-efficient fine-tuning technique that:
- Quantizes the base model to **4-bit precision** (reduces ~28GB → ~4GB VRAM)
- Adds small trainable **LoRA adapter layers** (~168MB)
- Achieves near full fine-tuning quality at a fraction of the cost

---

## 📁 Project Structure

```
Project1/
├── finetune.py                  # Main training script (QLoRA + SFTTrainer)
├── inference.py                 # Script for testing the fine-tuned model
├── generate_dataset.py          # Script to auto-generate Hinglish Q&A pairs
├── dataset.jsonl                # 500 Hinglish Q&A pairs (training data)
├── requirements.txt             # Python dependencies
├── run_finetuning_colab.ipynb   # Jupyter notebook for Google Colab execution
├── README.md                    # This file
└── mistral-finetuned/           # Output: trained adapter weights (after training)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── chat_template.jinja
```

---

## 🔄 Workflow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   LOCAL IDE      │     │     GITHUB       │     │  GOOGLE COLAB    │
│                  │────▶│                  │────▶│                  │
│ • Edit code      │     │ • Host code      │     │ • git pull       │
│ • Modify dataset │     │ • Version control│     │ • Train on T4 GPU│
│ • Push changes   │     │ • Collaboration  │     │ • Test & infer   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────────┐
                                                  │  HUGGING FACE    │
                                                  │                  │
                                                  │ • Save adapter   │
                                                  │ • Load anywhere  │
                                                  └──────────────────┘
```

1. **Local IDE** → Edit code, dataset, and configurations
2. **GitHub** → Push code to `https://github.com/9sushant/Project1.git`
3. **Google Colab** → Clone repo, install dependencies, train on T4 GPU
4. **Hugging Face Hub** → Save trained adapter permanently

---

## � Dataset

The dataset (`dataset.jsonl`) contains **500 Hinglish Q&A pairs** covering:

- **Gorakh Bodh** — Teachings of Guru Gorakhnath
- **Yoga & Chakras** — Kundalini, Nadi systems, Hatha Yoga
- **Vedanta Philosophy** — Advaita, Brahman, Maya, Atman
- **Nath Sampradaya** — Matsyendranath and Gorakhnath dialogues
- **Tantra & Rituals** — Mahanirvana Tantra, Bhairavi Chakra
- **Sanskrit Texts** — Shiva Stotras, Gayatri Mantra, Mahamrityunjaya
- **Yoga Vasistha** — Consciousness, creation, liberation philosophy
- **Ethics & Dharma** — Householder duties, Sannyas, Kali Yuga teachings

### Dataset Format
Each line in `dataset.jsonl` is a JSON object:
```json
{"user": "Gorakh Bodh text kya hai aur isme Shabad ka kya matlab bataya gaya hai?", "assistant": "Gorakh Bodh Guru Gorakhnath ki teachings ka collection hai jo 'Sandhya Bhasha' mein likha gaya hai..."}
```

### Dataset Generation
The `generate_dataset.py` script can auto-generate Hinglish Q&A pairs. Run it to create/expand the dataset:
```bash
python generate_dataset.py
```

---

## ⚙️ Training Configuration

### Final Optimized Settings (v3)

| Parameter | Value | Rationale |
|---|---|---|
| **LoRA Rank (r)** | 32 | Higher capacity for learning nuanced answers |
| **LoRA Alpha** | 64 | Balanced scaling (alpha = 2 × r) |
| **LoRA Dropout** | 0.01 | Minimal noise during training |
| **Target Modules** | q, k, v, o, gate, up, down proj | All attention + MLP layers |
| **Learning Rate** | 5e-5 | Balanced between speed and stability |
| **Epochs** | 7 | Multiple passes for memorization |
| **Batch Size** | 2 | Fits in T4 VRAM |
| **Gradient Accumulation** | 2 | Effective batch size = 4 |
| **Optimizer** | paged_adamw_8bit | Memory-efficient optimizer |
| **LR Scheduler** | Cosine | Smooth decay for convergence |
| **Warmup Steps** | 15 | Smooth training start |
| **Precision** | FP32 (no mixed precision) | Avoids BFloat16/GradScaler conflicts on T4 |
| **Trainable Parameters** | 83.8M / 7.3B (1.15%) | Only adapter weights are updated |

### Training Metrics Progression

| Run | Epochs | Steps | Final Loss | Token Accuracy | Time |
|---|---|---|---|---|---|
| Run 1 | 50 steps only | 50 | 3.689 | ~39% | 7 min |
| Run 2 | 3 | 189 | 1.686 | 64.5% | 26 min |
| Run 3 | 7 | 875 | ~1.2 | ~75%+ | ~60 min |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- Google Colab account (for T4 GPU)
- GitHub account
- Hugging Face account (for model hosting)

### Local Setup
```bash
git clone https://github.com/9sushant/Project1.git
cd Project1
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)
```
transformers
datasets
peft
trl
bitsandbytes
accelerate
torch
huggingface_hub
```

---

## ☁️ Training on Google Colab

### Quick Start
1. Open [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run these cells:

```python
# Cell 1: Install dependencies
!pip install -q transformers datasets peft trl bitsandbytes accelerate torch huggingface_hub

# Cell 2: Clone & enter project
!git clone https://github.com/9sushant/Project1.git
%cd Project1

# Cell 3: Train!
!python finetune.py
```

### If returning after training previously:
```python
# Pull latest changes
!cd Project1 && git pull

# Re-run training
!cd Project1 && python finetune.py
```

---

## 🧪 Inference & Testing

### Load & Chat with the Fine-tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load base model (quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto",
)

# Load fine-tuned adapter from Hugging Face
model = PeftModel.from_pretrained(base_model, "sushant2/mistral-hinglish-finetuned")
tokenizer = AutoTokenizer.from_pretrained("sushant2/mistral-hinglish-finetuned")
model.eval()

# Chat function
def chat(user_input):
    prompt = f"[INST] {user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# Test
print(chat("Gorakh Bodh text kya hai?"))
```

### Test Questions
```python
test_questions = [
    "Gorakh Bodh text kya hai aur isme Shabad ka kya matlab bataya gaya hai?",
    "Yogic anatomy ke anusar Shakti aur Shiva shareer mein kahan nivas karte hain?",
    "Mahamrityunjaya Mantra mein 'Tryambakam' shabd ka kya arth hai?",
    "Advaita Vedanta mein 'Ekam Evadvitiyam' ka kya matlab hai?",
]

for i, q in enumerate(test_questions, 1):
    print(f"\nQ{i}: {q}")
    print(f"A: {chat(q)}")
```

---

## 🤗 Saved Model on Hugging Face

The trained adapter weights are permanently saved at:

**🔗 [huggingface.co/sushant2/mistral-hinglish-finetuned](https://huggingface.co/sushant2/mistral-hinglish-finetuned)**

### Upload Adapter to Hugging Face (from Colab)
```python
from huggingface_hub import HfApi, login

login()  # Use a Write token from https://huggingface.co/settings/tokens

api = HfApi()
api.create_repo("sushant2/mistral-hinglish-finetuned", exist_ok=True)
api.upload_folder(
    folder_path="/content/Project1/mistral-finetuned",
    repo_id="sushant2/mistral-hinglish-finetuned",
)
```

### Load Saved Model (No Retraining Needed!)
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "sushant2/mistral-hinglish-finetuned")
```

---

## 🐛 Errors Encountered & Solutions

Throughout development, we encountered and resolved several issues:

### 1. `TypeError: TrainingArguments got unexpected keyword argument 'group_by_length'`
**Cause:** Newer version of `transformers` removed this argument.
**Fix:** Removed `group_by_length=True` from TrainingArguments.

### 2. `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`
**Cause:** T4 GPU doesn't natively support BFloat16, but `bitsandbytes` internally uses it.
**Fix:** Multi-step solution:
- Set `fp16=False` and `bf16=False` in TrainingArguments
- Added `prepare_model_for_kbit_training(model)` before applying PEFT
- Manually cast adapter parameters and buffers to float16
- Set `model.config.torch_dtype = torch.float16`

### 3. `TypeError: SFTTrainer got unexpected keyword argument 'max_seq_length'`
**Cause:** Newer version of `trl` changed SFTTrainer's constructor signature.
**Fix:** Removed `max_seq_length`, `dataset_text_field`, `tokenizer`, and `packing` arguments.

### 4. `warmup_ratio is deprecated`
**Cause:** `transformers` v5.2+ deprecated `warmup_ratio`.
**Fix:** Replaced with `warmup_steps=15`.

### 5. `JSON parse error: Missing a comma or '}' after an object member in row 102`
**Cause:** Two issues in `dataset.jsonl`:
- Line 103: Missing closing `}` brace
- Line 153: Two JSON objects concatenated on one line without a newline separator
**Fix:** Added missing brace and split the merged objects.

### 6. `NameError: name 'chat' is not defined`
**Cause:** Test cell was run before the cell that defines the `chat` function.
**Fix:** Run model loading + chat function definition cell first, then test.

### 7. `OutOfMemoryError: CUDA out of memory` (when uploading to HF)
**Cause:** Tried to reload the full model while training model was still in GPU memory.
**Fix:** Used `HfApi.upload_folder()` to upload files directly without loading the model.

### 8. `403 Forbidden: You don't have the rights to create a model`
**Cause:** Using GitHub username instead of HF username, and token lacked write permissions.
**Fix:** Used correct HF username (`sushant2`) and created a User Access Token with Write permissions.

### 9. `ModuleNotFoundError: No module named 'trl'`
**Cause:** Colab runtime restarted, clearing installed packages.
**Fix:** Re-run `!pip install -q transformers datasets peft trl bitsandbytes accelerate torch` before training.

---

## 📊 Results

### Training Progress (Run 2: 3 Epochs)
```
Epoch 0.16 → loss: 4.506, accuracy: 39.2%
Epoch 0.96 → loss: 2.270, accuracy: 56.4%
Epoch 1.91 → loss: 1.873, accuracy: 61.3%
Epoch 2.86 → loss: 1.686, accuracy: 64.6%
```

### Key Observations
- Loss dropped **63%** (4.506 → 1.686) over 3 epochs
- Token accuracy improved from **39% → 64.5%**
- Model learned Hinglish response style and domain-specific vocabulary
- Increasing epochs from 3 → 7 further improved accuracy

---

## 🤖 Next Steps

1. **Expand Dataset** — Add more Q&A pairs for better generalization
2. **Agentic AI Integration** — Plug the fine-tuned model into LangChain or AutoGen
3. **Evaluation Pipeline** — Automated comparison of model answers vs dataset answers
4. **Multi-turn Conversations** — Add conversation history support
5. **Deploy as API** — Host the model using Hugging Face Inference Endpoints

---

## 📚 Key Libraries Used

| Library | Purpose |
|---|---|
| `transformers` | Model loading, tokenization, training arguments |
| `peft` | LoRA adapter configuration and application |
| `trl` | SFTTrainer for supervised fine-tuning |
| `bitsandbytes` | 4-bit quantization (QLoRA) |
| `datasets` | Loading JSONL training data |
| `accelerate` | Distributed training and device management |
| `huggingface_hub` | Uploading trained model to HF Hub |

---

## 👤 Author

**Sushant Gaurav**
- GitHub: [@9sushant](https://github.com/9sushant)
- Hugging Face: [sushant2](https://huggingface.co/sushant2)

---

*Built as part of learning Fine-tuning & Agentic AI* 🧠
