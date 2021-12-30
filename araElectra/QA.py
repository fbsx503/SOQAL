from preprocess import ArabertPreprocessor
from transformers import pipeline


class QA:
    def __init__(self):
        self.prep = ArabertPreprocessor("aubmindlab/araelectra-base-discriminator")
        self.qa_pipe = pipeline("question-answering", model="wissamantoun/araelectra-base-artydiqa")

    def answerQuestion(self, question, context):
        print(self.tokenizer.tokenize(context))
        result = self.qa_pipe(question=self.prep.preprocess(question), context=context)
        print(result)

    def preprocess(self, text):
        return self.prep.preprocess(text)

    def unpreprocess(self, text):
        return self.prep.unpreprocess(text)
