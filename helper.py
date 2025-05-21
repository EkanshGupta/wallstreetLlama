import pandas as pd
import glob
import globals
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
import torch
from transformers import BitsAndBytesConfig
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
tqdm.pandas()

def merge_multiple_df(path):
    files = glob.glob(f"{path}reddit_finance_qa_*.jsonl")
    all_dfs = [pd.read_json(file, lines=True) for file in files]
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["instruction", "response"])
    merged_file = "reddit_finance_qa.jsonl"
    merged_df.to_json(merged_file, orient="records", lines=True)
    print(f"Merged {len(files)} files into {merged_file} with {len(merged_df)} unique QA pairs.")

    
def scrape_subs(subreddits):
    reddit = praw.Reddit(
    client_id=globals.client_id,
    client_secret=globals.client_secret,
    user_agent="scrape by /u/EkanshGupta"
    )
    limit_per_sub = 10000
    max_comments = 5
    sleep_interval = 1.0 
    qa_pairs = []

    for sub in subreddits:
        print(f"Scraping {sub}")
        subreddit = reddit.subreddit(sub)
        for i, post in enumerate(subreddit.hot(limit=limit_per_sub)):
            try:
                if post.selftext and len(post.selftext) > 30:
                    question = f"{post.title.strip()}\n{post.selftext.strip()}"
                else:
                    question = post.title.strip()

                post.comments.replace_more(limit=0)
                top_comments = sorted(post.comments, key=lambda x: x.score or 0, reverse=True)

                for comment in top_comments[:min(max_comments,len(top_comments))]:
                    response = comment.body.strip()
                    if len(response) > 30:
                        qa_pairs.append({
                            "subreddit": sub,
                            "instruction": question,
                            "response": response,
                            "post_id": post.id,
                            "score": post.score,
                            "url": post.url,
                            "timestamp": datetime.utcfromtimestamp(post.created_utc).isoformat()
                        })

            except Exception as e:
                print(f"[{sub}] Skipping post {post.id}: {e}")

            # Rate limiting
            time.sleep(sleep_interval)

            # Optional: progress update
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} posts scraped from r/{sub}...")

    df = pd.DataFrame(qa_pairs)
    df.to_json("reddit_finance_qa_1.jsonl", orient="records", lines=True)
    print(f"Saved {len(df)} Q&A pairs to reddit_finance_qa.jsonl")
    


def format_example(example):
    return {
        "text": f"### Question:\n{example['instruction']}\n\n### Answer:\n{example['response']}"
    }

def load_model(bnb_config, checkpoint_dir = "./lora_finetuned_llama2/checkpoint-1566/"):
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        if globals.model_name is None:
            raise ValueError("base_model_name must be provided when loading a LoRA/PEFT checkpoint.")
        print("Loading from saved checkpoint")
        base_model = AutoModelForCausalLM.from_pretrained(globals.model_name, quantization_config=bnb_config, device_map="auto")
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(globals.model_name, quantization_config=bnb_config, device_map="auto")
    return model

def prepare_data_and_train():
    data = load_dataset("json", data_files=globals.qa_data_path)["train"]
    data = data.map(format_example)
    data = data.train_test_split(test_size=0.05)
    model_name = globals.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=globals.token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

    # model = load_model(bnb_config)
    model = AutoModelForCausalLM.from_pretrained(globals.model_name, quantization_config=bnb_config, device_map="auto")

    if not isinstance(model, PeftModel):
        lora_config = LoraConfig(r=8,lora_alpha=32,lora_dropout=0.1,bias="none",task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
        model = get_peft_model(model, lora_config)

    #This config does not run out of memory
    training_args = TrainingArguments(
        output_dir="./lora_finetuned_llama2",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        label_names=["labels"]
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        formatting_func=lambda ex: ex["text"],
        args=training_args
    )

    trainer.train()
    
    
def generate(model, tokenizer, prompt, max_tokens=512):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    if full_output.startswith(prompt.strip()):
        return full_output[len(prompt):].strip()
    else:
        return full_output.strip()

    # generated = output[0][inputs["input_ids"].shape[1]:] 
    # answer = tokenizer.decode(generated, skip_special_tokens=True)
    # return answer

def format_prompt(example):
    return {
        "prompt": f"""Here is a Reddit post: {example['instruction'].strip()} Leave a helpful and insightful comment or answer below:""",
        "response": example["response"]
    }
    
def evaluate_model():
    tokenizer = AutoTokenizer.from_pretrained(globals.model_name, token=globals.token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    dataset = load_dataset("json", data_files=globals.qa_data_path)["train"]
    sampled = dataset.shuffle(seed=42).select(range(globals.num_questions))
    sampled = sampled.map(format_prompt)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(globals.model_name, quantization_config=bnb_config,device_map="auto")
    print("Base model loaded. Generating responses...")
    sampled_df = pd.DataFrame(sampled)
    print("Sampled prompts converted to df")
    sampled_df["base_answer"] = sampled_df["prompt"].progress_apply(lambda p: generate(base_model, tokenizer, p))
    print("Loading LoRA model")
    lora_model = PeftModel.from_pretrained(base_model, globals.finetuned_path)
    print("LoRA model loaded. Generating responses...")
    sampled_df["lora_answer"] = sampled_df["prompt"].progress_apply(lambda p: generate(lora_model, tokenizer, p))
    sampled_df["ground_truth"] = sampled_df["response"]  
    sampled_df[["prompt", "ground_truth", "base_answer", "lora_answer"]].to_csv(globals.comparison_output, index=False)
    print(f"Saved comparison to {globals.comparison_output}")

def embed_qa_faiss():
    globals.qa_file = "reddit_finance_qa.jsonl" 
    index_output_dir = "faiss_index"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Small + fast
    
    qa_pairs = []
    with open(qa_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            question = obj.get("instruction", "").strip()
            answer = obj.get("response", "").strip()
            if question and answer:
                qa_pairs.append(f"Q: {question}\nA: {answer}")
    
    print(f"Loaded {len(qa_pairs)} QA pairs.")
    
    print("Generating embeddings...")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(qa_pairs, show_progress_bar=True, convert_to_numpy=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    os.makedirs(index_output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_output_dir, "index.faiss"))
    with open(os.path.join(index_output_dir, "qa_corpus.pkl"), "wb") as f:
        pickle.dump(qa_pairs, f)
    
    print(f"Saved FAISS index and corpus to: {index_output_dir}")


class InferRAG:
    def __init__(self):
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.generation_model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use any LLM available to you
        self.index_dir = "faiss_index"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.top_k = 5
        
        self.index = faiss.read_index(f"{self.index_dir}/index.faiss")
        with open(f"{self.index_dir}/qa_corpus.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(self.embedding_model_name)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        
        print("Loading generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.generation_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.generation_model_name, device_map="auto",quantization_config=bnb_config)
        self.model.eval()
        
    def rag_generate(self, query, max_new_tokens=512):
        query_emb = self.embed_model.encode([query])
        _, indices = self.index.search(np.array(query_emb), self.top_k)
        retrieved_context = [self.corpus[i] for i in indices[0]]
        context = "\n\n".join(retrieved_context)
        prompt = f"""You are a helpful assistant.
    
    Use the following Reddit answers to help answer the user's question.
    
    {context}
    
    Question: {query}
    Answer:"""
        print("The prompt given to the LLM was:\n")
        print("\n###########################################################")
        print(prompt)
        print("###########################################################\n")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=globals.max_seq_len, do_sample=True, temperature=0.7)
        generated = outputs[0][inputs["input_ids"].shape[1]:] 
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        print("The answer given by the LLM is:\n###########################################################")
        print(answer)
        print("###########################################################\n")