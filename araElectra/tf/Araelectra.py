# %%
import torch
import json
import numpy as np
from . import qa_predict


class Araelectra():
    def __init__(self):
        self.model_electra = qa_predict.init_model()

    def get_answer(self, dataset):
        _res_electra = self.model_electra.predict(dataset)
        id = 0
        total_result = []
        for _ in _res_electra:
            total_result.append(_res_electra['q_'+str(id)])
        return total_result
