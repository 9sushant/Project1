import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    # ==========================
    # 1. Configuration Options
    # ==========================
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset_path = "dataset.jsonl"
    output_dir = "./mistral-finetuned"
    
    print(f"Loading Model: {model_name}")

    # ==========================
    # 2. Load and Prepare Dataset
    # ==========================
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    def format_prompt(examples):
        prompts = []
        for user_text, asst_text in zip(examples['user'], examples['assistant']):
            # Mistral Instruct uses [INST] tags
            prompts.append(f"<s>[INST] {user_text} [/INST] {asst_text} </s>")
        return {"text": prompts}

    dataset = dataset.map(format_prompt, batched=True)

    # ==========================
    # 3. BitsAndBytes Config (QLoRA)
    # ==========================
    # We use 4-bit precision to fit Mistral 7B into Colab's T4 GPU (~15GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # ==========================
    # 4. Load Model and Tokenizer
    # ==========================
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix padding issue for fp16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False # Disable caching for gradient checkpointing

    # ==========================
    # 5. LoRA Adapter Config
    # ==========================
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Apply PEFT model explicitly
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ==========================
    # 6. Training Arguments
    # ==========================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit", # Optimizer optimized for limited memory
        save_steps=50,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True, # Depending on GPU, bf16 might be better (A100), but fp16 for T4
        max_grad_norm=0.3,
        max_steps=50, # Low value for learning/demonstration purposes
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # disable wandb
    )

    # ==========================
    # 7. SFT Trainer
    # ==========================
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024, # Maximum length of input tokens
        tokenizer=tokenizer,
        args=training_args,
        packing=False
    )

    print("--- Starting SFT Training ---")
    trainer.train()

    # ==========================
    # 8. Save the Model
    # ==========================
    print("Saving the tuned adapter...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Adapter saved to {output_dir}")

if __name__ == "__main__":
    main()
