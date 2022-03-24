# %%
import torch
import json
import numpy as np
from . import qa_predict


class Araelectra():
    def __init__(self):
        self.model_electra = qa_predict.init_model()

    def get_answer(self, question, paragraph):
        _res_electra = self.model_electra.predict(question, paragraph)
        return _res_electra['q_0'][0]
