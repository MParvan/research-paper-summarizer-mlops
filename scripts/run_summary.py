# scripts/run_summary.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.model.inference import Summarizer
import mlflow
import os


def main(args, article):
    summarizer = Summarizer(model_name=args.model, max_length=args.max_length, min_length=args.min_length)
    summary = summarizer.summarize(article)

    # Save output
    os.makedirs("outputs", exist_ok=True)
    output_file = os.path.join("outputs", "generated_summary.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    # Log to MLflow
    mlflow.set_experiment("cli-arxiv-summary")
    with mlflow.start_run():
        mlflow.log_param("model", args.model)
        mlflow.log_param("max_length", args.max_length)
        mlflow.log_param("min_length", args.min_length)
        mlflow.log_text(article, "inputs/article.txt")
        mlflow.log_text(summary, "outputs/generated_summary.txt")
        mlflow.log_metric("input_length", len(article))
        mlflow.log_metric("output_length", len(summary))

    print("\n✅ Summary written to:", output_file)
    print("\n📄 Summary Preview:\n", summary[:500])


def extract_text_from_pdf(pdf_path):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize a research article using BART")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input .txt file")
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn", help="Model name")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--min_length", type=int, default=64)

    args = parser.parse_args()

    if args.input_path.endswith(".pdf"):
        article = extract_text_from_pdf(args.input_path)
    else:
        with open(args.input_path, "r", encoding="utf-16 LE") as f:
            article = f.read()

    main(args, article)
