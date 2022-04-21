from retriever.CustomRetriever import *
import pickle
import os.path



class retriever_cache:
    def __init__(self,retriever):
        self.retriever = retriever
        self.retriever_cache = {"changed": False}
        if isinstance(retriever, CustomRetriever):
            self.retriever_cache_path = 'retriever/docsCacheGoogle.txt'
        else:
            self.retriever_cache_path = 'retriever/docsCache.txt'
        print(self.retriever_cache_path)

        
    

    def get_topk_docs_scores_cache(self, question):
        if question in self.retriever_cache:
            return self.retriever_cache[question]
        else:
            docs, doc_scores = self.retriever.get_topk_docs_scores(question)
            self.retriever_cache[question] = [docs, doc_scores]
            self.retriever_cache["changed"] = True
        return docs, doc_scores

    def dumb_retirever_cache(self):
        if self.retriever_cache["changed"] is True:
            print("Saving retriever cache")
            file = open(self.retriever_cache_path, 'wb+')
            pickle.dump(self.retriever_cache, file)
            file.close()
            self.retriever_cache["changed"] = False
            print("Cache Saved")

    def load_retriever_cache(self):
        print("Loading retrievr cache...")
        if not os.path.exists(self.retriever_cache_path):
            self.retriever_cache["changed"] = False
            print("Cache file doesn't exist!")
            return
        dbfile = open(self.retriever_cache_path, 'rb')
        self.retriever_cache = pickle.load(dbfile)
        dbfile.close()
        self.retriever_cache["changed"] = False
        print("Cache Loaded")
