# %%
import torch
import json
import numpy as np
from . import qa_predict


class Araelectra():
    def __init__(self):
        self.model_electra = qa_predict.init_model()

    def get_answer(self, question, paragraph):
        question = question.lstrip().rstrip()
        _res_electra = qa_predict.predict(question, paragraph, self.model_electra)
        return _res_electra
