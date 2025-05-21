import praw
import pandas as pd
from datetime import datetime
import time

client_id = ""
client_secret = ""
model_name = "meta-llama/Llama-2-7b-hf"
token = ""

base_model_path = "meta-llama/Llama-2-7b-hf"
finetuned_path = "./lora_finetuned_llama2/checkpoint-3130/"
qa_data_path = "reddit_finance_qa.jsonl"
comparison_output = "comparison.csv"
num_questions = 100
max_seq_len = 512
