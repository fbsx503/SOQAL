from transformers import T5Tokenizer, MT5ForConditionalGeneration
import torch
import numpy as np
from scipy.special import softmax

class BertRanker:
    def __init__(self):
        self.model_name = 'unicamp-dl/mt5-base-mmarco-v2'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)


    def rank(self, question , article):
        result = len(article.split())
        if result > 490:
            return 1000000000
        inputs = self.tokenizer(article, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(question, return_tensors="pt")
        outputs = self.model(**inputs, labels=labels["input_ids"])
        logits = outputs["logits"]
        distributions = torch.softmax(logits, -1)
        decoder_input_ids = labels["input_ids"].unsqueeze(-1)
        batch_probs = torch.gather(distributions, 2, decoder_input_ids).squeeze(-1)
        masked_log_probs = torch.log10(batch_probs) * labels["attention_mask"]
        scores = torch.sum(masked_log_probs, 1)
        return scores.item()

