# %%
import torch
import json
import numpy as np
from . import qa_predict


class araelectra():
    def __init__(self):
        self.model_electra = qa_predict.init_model()

    def get_answer(self, dataset):
        _res_electra = qa_predict.predict(dataset, self.model_electra)
        id = 0
        total_result = []
        for _ in _res_electra['squadv1']:
            total_result.append(_res_electra['squadv1']['q_'+str(id)])
            id += 1
        return total_result
