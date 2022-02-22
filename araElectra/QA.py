from .preprocess import ArabertPreprocessor
from transformers import pipeline


class QA:
    def __init__(self):
        self.prep = ArabertPreprocessor("aubmindlab/araelectra-base-discriminator")
        self.qa_pipe = pipeline("question-answering", model="/home/aymanm419/run2")

    def answerQuestion(self, question, context):
        result = self.qa_pipe(question=self.prep.preprocess(question), context=context)
        return result

    def preprocess(self, text):
        return self.prep.preprocess(text)

    def unpreprocess(self, text):
        return self.prep.unpreprocess(text)
