from .preprocess import ArabertPreprocessor
from transformers import pipeline


class QA:
    def __init__(self, model):
        self.prep = ArabertPreprocessor(model)

    def answerQuestion(self, question, context):
        result = ""
        return result

    def preprocess(self, text):
        return self.prep.preprocess(text)

    def unpreprocess(self, text):
        return self.prep.unpreprocess(text)
