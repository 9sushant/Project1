# Project 1: Fine-tuning Mistral with QLoRA

Welcome to **Project 1**! This project is designed to help you learn Fine-tuning and Agentic AI. We'll be using the Mistral 7B model and the memory-efficient QLoRA method on a custom Q&A dataset.

## 🎯 Workflow Overview

1. **Local IDE (You are here!)**
   Review and modify your Python scripts, Custom Dataset (`dataset.jsonl`), and configurations.
2. **Push to GitHub**
   Commit your changes locally and push this codebase to a new repository on your GitHub.
3. **Run in Cloud (Google Colab)**
   Upload `run_finetuning_colab.ipynb` to Google Colab, or open it directly via GitHub integration, and let the cloud GPU do the heavy lifting of training!

## 📁 Files Included

*   `finetune.py`: The main Python script that handles data loading, model quantization (QLoRA), and the `SFTTrainer` setup.
*   `dataset.jsonl`: Your custom Question & Answer dataset in JSON Lines format. Modify this to add your own data!
*   `requirements.txt`: The Python dependencies needed to run the fine-tuning script.
*   `run_finetuning_colab.ipynb`: A Jupyter Notebook template making it seamless to clone your repo onto Google Colab and execute the fine-tuning script.

## 🚀 Step-by-Step Guide

### Step 1: Prepare the Dataset Locally
Open `dataset.jsonl` in your IDE and add your custom Q&A pairs. Follow the format:
```json
{"user": "Question here", "assistant": "Answer here"}
```

### Step 2: Push to GitHub
If you haven't already:
```bash
git init
git add .
git commit -m "Initial commit for Mistral Fine-Tuning Project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

### Step 3: Train on Google Colab!
1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **File > Open Notebook**.
3. Choose the **GitHub** tab, grant access, and select your newly created repository and the `run_finetuning_colab.ipynb` file.
4. **Important:** Change the Runtime type to use a T4 GPU (Runtime > Change runtime type > Hardware accelerator > T4 GPU).
5. Update the notebook's GitHub URL to point to your repository.
6. Run all cells! The script will download Mistral, apply QLoRA, fine-tune it on your `dataset.jsonl`, and save the adapters in the `mistral-finetuned` folder.

## 🤖 Next Steps (Agentic AI)
Once your model is fine-tuned, you can load these adapters and plug them into an Agent framework (like LangChain or AutoGen) to test how well it performs complex, multi-step problem solving!
