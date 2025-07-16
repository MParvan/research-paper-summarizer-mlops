# src/model/inference.py

from transformers import pipeline
import torch


class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", max_length=256, min_length=64, device=None):
        self.device = 0 if torch.cuda.is_available() else -1 if device is None else device
        self.max_length = max_length
        self.min_length = min_length

        self.pipeline = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=self.device
        )

    def summarize(self, text: str) -> str:
        if len(text) > 1024:
            text = text[:1024]  # Truncate to avoid token limit issues
        result = self.pipeline(text, max_length=self.max_length, min_length=self.min_length, do_sample=False)
        return result[0]['summary_text']
