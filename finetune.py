import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.config.use_cache = False # Disable caching for gradient checkpointing
    model.config.torch_dtype = torch.float16 # Hard override to prevent PEFT from inferring BFloat16

    # Cleanse any lingering bfloat16 buffers
    for buffer in model.buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)

    # ==========================
    # 5. LoRA Adapter Config
    # ==========================
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.01,         # Reduced from 0.05 — less noise during training
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Prepare model for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply PEFT model explicitly
    model = get_peft_model(model, lora_config)
    
    # Force float16 for all adapter parameters to fix T4 bfloat16 errors
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
            
    model.print_trainable_parameters()

    # ==========================
    # 6. Training Arguments
    # ==========================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,   # More frequent updates (was 4)
        optim="paged_adamw_8bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=5e-5,              # Raised from 2e-5 for faster learning
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        num_train_epochs=7,              # 7 full passes (was 3) — key for memorization
        warmup_steps=15,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=2,
    )

    # ==========================
    # 7. SFTTrainer Update (v0.8.0+)
    # ==========================
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
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
