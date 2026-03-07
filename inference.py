import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_finetuned_model(
    base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    adapter_path="./mistral-finetuned"
):
    """Load the base model with QLoRA adapters merged in."""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"Loading fine-tuned adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Model ready!\n")
    return model, tokenizer


def chat(model, tokenizer, question, max_new_tokens=256):
    """Send a question to the fine-tuned model and get a response."""
    
    # Format using Mistral Instruct template
    prompt = f"<s>[INST] {question} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15
        )
    
    # Decode only the new tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    model, tokenizer = load_finetuned_model()
    
    # =============================
    # Test Questions
    # =============================
    test_questions = [
        "What is Agentic AI?",
        "How does QLoRA work for fine-tuning?",
        "What are the benefits of using Mistral models?",
        "Why should I use Google Colab for training?",
        "What is the typical workflow for developing an AI fine-tuning project?",
        # Try a question NOT in the training data to test generalization
        "What is the difference between LoRA and full fine-tuning?",
    ]

    print("=" * 60)
    print("  TESTING FINE-TUNED MISTRAL MODEL")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"  Q{i}: {question}")
        print(f"{'─' * 60}")
        answer = chat(model, tokenizer, question)
        print(f"  A: {answer}")
    
    # =============================
    # Interactive Chat Loop
    # =============================
    print(f"\n{'=' * 60}")
    print("  INTERACTIVE CHAT (type 'quit' to exit)")
    print(f"{'=' * 60}")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_input:
            continue
        response = chat(model, tokenizer, user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
