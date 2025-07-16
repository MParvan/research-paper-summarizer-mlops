# Step 1: Imports
import mlflow
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Step 2: Load Dataset
dataset = load_dataset("ccdv/arxiv-summarization", split="train[:1%]")  # Small sample for testing
sample = dataset[0]
article = sample["article"]
reference_summary = sample["abstract"]

# Step 3: Load Model (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Optional: Truncate for demo due to input size limits (BART has ~1024 tokens limit)
article_input = article[:1024]

# Step 4: Run Inference
generated = summarizer(article_input, max_length=256, min_length=64, do_sample=False)
generated_summary = generated[0]["summary_text"]

# Step 5: Log to MLflow
mlflow.set_experiment("arxiv-bart-summarization")

with mlflow.start_run():
    mlflow.log_param("model_name", "facebook/bart-large-cnn")
    mlflow.log_param("max_length", 256)
    mlflow.log_param("min_length", 64)

    mlflow.log_text(article_input, "inputs/article.txt")
    mlflow.log_text(reference_summary, "inputs/reference_summary.txt")
    mlflow.log_text(generated_summary, "outputs/generated_summary.txt")

    mlflow.log_metric("input_length", len(article_input))
    mlflow.log_metric("output_length", len(generated_summary))

    print("Summary logged to MLflow.")
    print("Generated Summary:\n", generated_summary)
