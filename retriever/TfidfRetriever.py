# -*- coding: utf-8 -*-
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.arlstem import ARLSTem
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pickle
from gensim.summarization.bm25 import BM25
from numpy import dot, array
from scipy import sparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--topk", type=int, default=10, help="number of documents retriever should return")
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', required=True)
parser.add_argument('-o', '--output-dir', help='Where to place the retrivers', required=True)



class TfidfRetriever:
    SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"'

    def __init__(self, docs, k, ngrams, vectorizer=None, tfidf_matrix=None):
        self.k = k  # number of documents to return
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = ARLSTem()
        self.docs = docs
        self.stopwords = stopwords.words('arabic')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), norm=None, stop_words=self.stopwords)
        if tfidf_matrix is None or vectorizer is None:
            docs_stemmed = self.docs_stem()
            self.tfidf_matrix = self.vectorizer.fit_transform(docs_stemmed)
        else:
            self.vectorizer = vectorizer
            self.tfidf_matrix = tfidf_matrix

    def docs_stem(self):
        docs_stemmed = []
        for d in self.docs:
            docs_stemmed.append(self.stem_string(d))
        return docs_stemmed

    def stem_string(self, str):
        str_tokens = self.tokenizer.tokenize(str)
        str_processed = ""
        for token in str_tokens:
            has_symbol = False
            for s in self.SYMBOLS:
                if s in token:
                    has_symbol = True
                    break
            if not has_symbol:
                str_processed += token + " "
        return str_processed

    def get_topk_docs_scores(self, query):
        """
        :param query: question as string
        :return: the top k articles with each of their paragraphs seperated by '###' as python list of strings
        """
        qeury = self.stem_string(query)
        query_tfidf = self.vectorizer.transform([query])
        similarities_raw = linear_kernel(self.tfidf_matrix, query_tfidf)
        similarities = []
        for s in similarities_raw:
            similarities.append(s[0])
        indices_sorted = np.argsort(similarities)[::-1]  # reverse order
        top_docs = []
        docs_scores = []
        i = 0
        while i < min(self.k, len(self.docs)):
            doc = self.docs[indices_sorted[i]]
            top_docs.append(doc)
            docs_scores.append(similarities[indices_sorted[i]])
            i += 1
        norm_cst = np.sum(np.asarray(docs_scores))
        docs_scores = np.asarray(docs_scores)
        docs_scores = docs_scores / norm_cst
        return top_docs, docs_scores

    def get_topk_docs(self, query):
        """
        :param query: question as string
        :return: the top k articles with each of their paragraphs seperated by '###' as python list of strings
        """
        qeury = self.stem_string(query)
        query_tfidf = self.vectorizer.transform([query])
        similarities_raw = linear_kernel(self.tfidf_matrix, query_tfidf)
        similarities = []
        for s in similarities_raw:
            similarities.append(s[0])
        indices_sorted = np.argsort(similarities)[::-1]  # reverse order
        top_docs = []
        scores = []
        i = 0
        while i < min(self.k, len(self.docs)):
            doc = self.docs[indices_sorted[i]]
            top_docs.append(doc)
            i += 1
        norm_cst = np.sum(np.asarray(scores))
        return top_docs




class TfidfRetriever_sys:
    SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"'
    def __init__(self, docs, k, ngrams, vectorizer=None, tfidf_matrix=None):
        self.k = k  # number of documents to return
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = ARLSTem()
        self.docs = docs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), norm=None)
        if tfidf_matrix is None or vectorizer is None:
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        else:
            self.vectorizer = vectorizer
            self.tfidf_matrix = tfidf_matrix


    def get_topk_docs(self, query):
        """
        :param query: question as string
        :return: the top k articles with each of their paragraphs seperated by '###' as python list of strings
        """
        query_tfidf = self.vectorizer.transform([query])
        similarities_raw = linear_kernel(self.tfidf_matrix, query_tfidf)
        similarities = []
        for s in similarities_raw:
            similarities.append(s[0])
        indices_sorted = np.argsort(similarities)[::-1]  # reverse order
        top_docs = []
        i = 0
        while i < min(self.k, len(self.docs)):
            doc = self.docs[indices_sorted[i]]
            top_docs.append(doc)
            i += 1
        return top_docs



class HierarchicalTfidf:
    def __init__(self, base_retriever, k1, k2):
        self.r = base_retriever
        self.r.k = k1
        self.k = k2

    def get_topk_docs_scores(self, query):
        docs = self.r.get_topk_docs(query)
        pars = []
        for doc in docs:
            ps = doc.split("###")
            for p in ps:
                pars.append(p)
        r2 = TfidfRetriever(pars, self.k, 2)
        top_docs, docs_scores = r2.get_topk_docs_scores(query)
        return top_docs, docs_scores

    def get_topk_docs(self, query):
        docs = self.r.get_topk_docs(query)
        r2 = TfidfRetriever_sys(docs, self.k, 2)
        top_docs = r2.get_topk_docs(query)
        return top_docs


class bm25:
    SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"'

    def __init__(self, docs, k):
        self.k = k
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = ARLSTem()
        self.docs = self.docs_stem(docs)
        self.stopwords = stopwords.words('arabic')

    def docs_stem(self, docs):
        docs_stemmed = []
        for d in docs:
            docs_stemmed.append(self.stem_string(d))
        return docs_stemmed

    def stem_string(self, str):
        str_tokens = self.tokenizer.tokenize(str)
        str_processed = ""
        for token in str_tokens:
            has_symbol = False
            for s in self.SYMBOLS:
                if s in token:
                    has_symbol = True
                    break
            if not has_symbol:
                str_processed += token + " "
        return str_processed

    def get_topk_docs_scores(self, question):
        question = self.stem_string(question)
        tok_corpus = [s.split() for s in self.docs]
        bm25 = BM25(tok_corpus)
        query = question.split()
        scores = bm25.get_scores(query)

        best_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        final_docs = []
        final_scores = []
        for i, b in enumerate(best_docs):
            final_docs.append(self.docs[b])
            final_scores.append(scores[b])
        return final_docs, final_scores


def build_tfidfretriever(wiki_path, output_path, k):
    wiki_data = pickle.load(open(wiki_path, "rb"))
    docs = []
    i = 0
    for art, pars in wiki_data.items():
        article_text = ""
        for p in pars:
            article_text += p + "### "
        docs.append(article_text)
        i += 1
    print("finished building documents")
    r = bm25(docs, k)
    pickle.dump(r, open(output_path + "/bm25retriever.p", "wb"))



def main():
    args = parser.parse_args()
    build_tfidfretriever(args.wiki_path, args.output_dir, args.topk)

if __name__ == "__main__":
    main()
