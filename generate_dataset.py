import json
import random

def generate_hinglish_dataset(num_samples=600):
    dataset = []

    # Topics: AI, Fine-tuning, Agentic AI, Coding, Hardware
    
    questions_ai = [
        "AI kya hota hai?",
        "Bhai, ye Artificial Intelligence kaise kaam karta hai?",
        "Machine Learning ka basic concept samjha do.",
        "Deep Learning aur Machine Learning mein kya difference hai?",
        "Neural Networks kya hote hain?",
        "LLMs (Large Language Models) kya hain?",
        "Generative AI kis chidiya ka naam hai?",
        "Ye ChatGPT ke peeche konsi technology hai?",
        "Kya AI hamaari jobs kha jayega?",
        "AI reasoning kaise karta hai?"
    ]
    
    answers_ai = [
        "AI ya Artificial Intelligence ek aisi field hai jahan hum computers ko human jaisi soch aur samajh dete hain.",
        "Simple bhasha mein, AI computers ko data se sikhne aur khud decisions lene ki ability deta hai.",
        "Machine Learning AI ka ek hissa hai jismein bina explicitly program kiye system patterns sikh leta hai.",
        "Deep Learning actually ML ka advanced version hai jo human brain ke neural networks se inspired hai.",
        "Neural Networks layers mein interconnected nodes hote hain jo complex patterns detect karte hain.",
        "LLMs bohot bade text data pe train kiye gaye models hain, jaise Mistral aur GPT, jo human-like text generate karte hain.",
        "Generative AI naya content create kar sakta hai, jaise text, images ya code, existing data se seekh kar.",
        "ChatGPT Transformers architecture par based ek LLM hai jo human jaisi baatein kar sakta hai.",
        "AI jobs replace karne se zyada unhe transform karega, isliye AI tools seekhna important hai.",
        "AI reasoning math aur probability ka use karke logical connections banata hai, bilkul human jaisa lagta hai par asal mein statistics hai."
    ]

    questions_finetuning = [
        "Fine-tuning kya hota hai LLMs mein?",
        "Kya mujhe apne custom data pe model train karna chahiye?",
        "LoRA aur QLoRA mein kya farq hai?",
        "Model ko fine-tune karne ke liye kitna VRAM chahiye?",
        "PEFT kya hota hai?",
        "Bhai QLoRA kyu use karein?",
        "Mistral 7B ko fine-tune kaise karein?",
        "Epochs kya hote hain training mein?",
        "Overfitting kya hai aur isse kaise bachein?",
        "Learning rate kaise set karte hain fine-tuning mein?"
    ]

    answers_finetuning = [
        "Fine-tuning ka matlab hai pre-trained model ko apne specific use-case ya custom data ke hisaab se aur zyada smart banana.",
        "Haan bilkul, agar tumhe model se specific kaam karwana hai jaise legal advice ya coding, toh custom data pe fine-tune karna best hai.",
        "LoRA sirf kuch weights ko update karta hai memory bachane ke liye, jabki QLoRA weights ko 4-bit mein compress (quantize) karke memory aur bhi bacha leta hai.",
        "Agar full fine-tune karna hai toh 40GB+ VRAM chahiye, par QLoRA use karoge toh 12-16GB VRAM (jaise Colab T4) mein bhi kaam ho jayega.",
        "PEFT (Parameter-Efficient Fine-Tuning) ek technique hai jismein model ke sirf kuch parameters ko train kiya jata hai bajaye pure model ke.",
        "QLoRA isliye use karte hain taaki consumer GPUs (jaise RTX 3060/4090 ya Colab T4) pe bade models ko easily fine-tune kiya ja sake.",
        "Mistral 7B ko fine-tune karne ke liye Hugging Face ka `trl` library, QLoRA, aur custom JSONL dataset ka use karna sabse aasan hai.",
        "Ek epoch ka matlab hota hai ki tumhara model poore dataset ko kitni baar dekh chuka hai training ke dauran.",
        "Overfitting tab hota hai jab model training data ko raatt leta hai aur naye data pe fail ho jata hai. Isse bachne ke liye dropout and early stopping use karte hain.",
        "Learning rate kaafi chota hona chahiye, jaise 2e-4 ya 1e-5, taaki model dhire dhire seekhe aur apne purane weights ko poori tarah bhool na jaye."
    ]

    questions_agentic = [
        "Agentic AI naya buzzword hai kya?",
        "Agentic AI normal AI se kaise alag hai?",
        "Kya AI agents khud code execute kar sakte hain?",
        "AI agent ko tools kaise diye jaate hain?",
        "AutoGPT ya LangChain ka Agentic AI mein kya role hai?",
        "Agentic reasoning kaise kaam karti hai?",
        "ReAct prompting kya hota hai?",
        "Ek AI agent internet browse kar sakta hai?",
        "Kya Agentic AI future hai?",
        "Agents memory kaise maintain karte hain?"
    ]

    answers_agentic = [
        "Agentic AI sirf buzzword nahi hai, yeh AI ka next evolution hai jahan models autonomously action le sakte hain.",
        "Normal AI sirf chat ya text generate karta hai. Agentic AI goals set karta hai, internet search karta hai aur external tools use karke tasks complete karta hai.",
        "Haan, Agentic systems (jaise AutoGen ya Code Interpreter) Python scripts likh kar unko securely execute bhi kar sakte hain.",
        "Agents ko functions aur APIs ke descriptions diye jaate hain. Jab unko zaroorat hoti hai, woh in functions ko call karne ka text generate karte hain.",
        "LangChain aur AutoGPT frameworks hain jo AI models ko tools, memory aur goal-oriented workflows banane mein help karte hain.",
        "Agentic reasoning mein AI pehle step-by-step plan banata hai, fir action leta hai, aur uske result ko observe karke apna next step decide karta hai.",
        "ReAct (Reasoning and Acting) ek prompting technique hai jahan model pehle sochta hai ('Thought:'), fir action leta hai, aur finally output nikalta hai.",
        "Bilkul, agar agent ke paas web-search ya browser tool ka access ho, toh woh real-time data nikal kar answers de sakta hai.",
        "Haan, Agentic AI pichhle chat-based AI (jaise ChatGPT) ko replace karke autonomous task-solving systems ki taraf badhne ka raasta hai.",
        "Agents long-term memory ke liye vector databases (jaise Pinecone ya ChromaDB) ka use karte hain aur short-term memory ke liye chat history pass karte hain."
    ]

    questions_coding = [
        "Python hi kyu AI ke liye?",
        "Hugging Face kya hai?",
        "Colab T4 GPU kaisa hai AI training ke liye?",
        "Transformers library ka kya use hai?",
        "PyTorch aur TensorFlow mein kya better hai?",
        "JSONL dataset format hi kyu use karte hain fine-tuning mein?",
        "Out of Memory (OOM) error kaise theek karein PyTorch mein?",
        "Batch size kitna rakhna chahiye fine-tuning mein?",
        "Gradient Accumulation kis kaam aata hai?",
        "Model quantization kya hota hai?"
    ]

    answers_coding = [
        "Python learn karne mein aasan hai aur isme PyTorch, TensorFlow, Pandas jaisi powerful AI/ML libraries ka ecosystem best hai.",
        "Hugging Face AI models, datasets aur libraries (jaise transformers aur peft) ka ek open-source hub hai, basically AI ka GitHub.",
        "Colab T4 (15GB VRAM) basic fine-tuning aur inference ke liye badhiya aur free hai, par bade models ke liye L4 ya A100 lagta hai.",
        "Transformers library pre-trained models (LLMs) ko load karne, train karne aur use karne ke liye simple aur powerful API deti hai.",
        "Ajj kal research aur production dono mein PyTorch zyada popular hai kyunki iska dynamic computational graph aur Pythonic feel better hai.",
        "JSONL (JSON Lines) mein har line ek independent JSON object hoti hai, jisse memory-efficient tarike se large datasets line-by-line read kiye ja sakte hain.",
        "OOM errors avoid karne ke liye batch_size kam karo, gradient accumulation steps badhao, ya QLoRA/mixed precision (fp16) ka use karo.",
        "Batch size GPU VRAM pe depend karta hai. Colab T4 pe 7B model ke liye batch size 1 ya 2 rakha jata hai.",
        "Agar VRAM kam hai toh hum chote batch sizes pe gradients calculate karte hain aur unhe accumulate karke ek saath update karte hain. Isse large batch training simulate hoti hai.",
        "Quantization model ke weights ki precision (jaise 16-bit float) ko kam karke 8-bit ya 4-bit mein badal deta hai, jisse model ka size aur RAM requirement aadhi ya usse bhi kam ho jati hai."
    ]

    greetings_q = ["Bhai kaisa hai?", "Hello Agent!", "Hi", "Kaise ho tum?", "Tumhara kya naam hai?"]
    greetings_a = [
        "Main badhiya hoon! Main aapka AI assistant hoon, bataiye main aapki kya madad kar sakta hoon Agentic AI aur coding mein?",
        "Hello! Main ek fine-tuned AI agent hoon. Puchhiye apne sawal!",
        "Hi there! Hinglish ya English, dono mein puchiye apne tech aur AI questions.",
        "Main bilkul theek hoon! Machine learning aur fine-tuning me kuch sikhna hai aaj?",
        "Mera koi insaani naam nahi hai, main ek Mistral-based Agentic AI hoon jisko aapne fine-tune kiya hai!"
    ]

    categories = [
        (questions_ai, answers_ai),
        (questions_finetuning, answers_finetuning),
        (questions_agentic, answers_agentic),
        (questions_coding, answers_coding)
    ]

    # Add the current 5 examples from original dataset.jsonl just to be safe
    dataset.append({"user": "What is Agentic AI?", "assistant": "Agentic AI refers to artificial intelligence systems designed to autonomously pursue goals, make real-world decisions, and take actions with minimal human intervention. Unlike traditional conversational models that only answer queries, Agentic AI can formulate plans, use external tools like web browsers or terminals, and adapt to changing environments to solve complex problems."})
    dataset.append({"user": "How does QLoRA work for fine-tuning?", "assistant": "QLoRA (Quantized Low-Rank Adaptation) is a highly efficient fine-tuning technique that reduces memory usage significantly. It works by quantizing a large language model to 4-bit precision using NormalFloat4 (NF4) while freezing its original weights. Then, small, trainable low-rank adapter matrices are added to specific layers of the model. This allows you to fine-tune massive models like Mistral on a single consumer GPU without compromising performance."})
    dataset.append({"user": "What are the benefits of using Mistral models?", "assistant": "Mistral models, particularly the 7B parameter version, offer exceptional performance for their size. They often outperform larger models in reasoning, coding, and comprehension tasks due to advanced architectural choices like Grouped-Query Attention (GQA) for faster inference and Sliding Window Attention (SWA) for handling longer contexts efficiently."})
    dataset.append({"user": "Why use Google Colab for training instead of a local machine?", "assistant": "Training Large Language Models requires substantial GPU VRAM, which local consumer machines often lack. Google Colab provides free or low-cost access to powerful cloud GPUs like the T4, L4, or A100. By using Colab, you can offload the heavy computational workload of fine-tuning, keeping your local machine free for development, while a cloud GPU handles the memory-intensive QLoRA training process."})
    dataset.append({"user": "What is the typical workflow for developing an AI fine-tuning project?", "assistant": "A typical professional workflow involves writing and testing your code logic locally in an IDE, such as VS Code or Cursor. Then, you push the clean codebase and dataset to a version control system like GitHub. Finally, you clone the repository onto a cloud GPU provider (e.g., Google Colab, RunPod, or AWS) and execute the training script there. This ensures code versioning, reproducible environments, and separation of development from heavy computation."})


    # We need ~600 records. We will synthetically combine and rephrase slightly.
    # To hit 600 quickly, we will create templates and random modifiers.
    
    modifiers_q = [
        "Bhai bataiye, ",
        "Dost, ",
        "Can you explain, ",
        "Mujhe samjhao ki ",
        "Ek baat batao, ",
        "Thoda detail me batao ki ",
        "Kya tumhe pata hai ",
        "",
        "Quick answer chahiye: "
    ]
    
    modifiers_a = [
        "Dekho bhai, ",
        "Sure! ",
        "Bohot simple hai: ",
        "Asaan bhasha me kahu toh, ",
        "Haan bilkul. ",
        "Technical terms mein: ",
        "",
        "To the point: "
    ]

    for _ in range(num_samples):
        # Decide if it's a greeting or a tech question
        if random.random() < 0.05:
            q = random.choice(greetings_q)
            a = random.choice(greetings_a)
        else:
            cat_q, cat_a = random.choice(categories)
            idx = random.randint(0, len(cat_q)-1)
            
            q_mod = random.choice(modifiers_q)
            a_mod = random.choice(modifiers_a)
            
            # Lowercase the first letter sometimes to blend with modifier
            base_q = cat_q[idx]
            base_a = cat_a[idx]
            
            if q_mod:
                base_q = base_q[0].lower() + base_q[1:]
            
            q = f"{q_mod}{base_q}"
            a = f"{a_mod}{base_a}"
        
        # Add slight variations to prevent exact duplicates if preferred
        dataset.append({
            "user": q.strip(),
            "assistant": a.strip()
        })

    # Shuffle dataset
    random.shuffle(dataset)

    # Write to JSONL
    with open("dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(dataset)} examples in dataset.jsonl")

if __name__ == "__main__":
    generate_hinglish_dataset(1000)
