from .GoogleSearchRetriever import ApiGoogleSearchRetriever
from .TfidfRetriever import HierarchicalTfidf

class CustomRetriever:
    def __init__(self, base_retriever, wikipedia, k1, k2):
        base_retriever.k = k1
        self.k = k2
        self.gret = ApiGoogleSearchRetriever(wikipedia, k2)
        self.ret = HierarchicalTfidf(base_retriever, k1, k2)

    def get_topk_docs_scores(self, query):
        docs, scores = self.gret.get_topk_docs_scores(query)
        if docs is None or len(docs) < 5:
            docs, scores = self.ret.get_topk_docs_scores(query)
        return docs, scores
