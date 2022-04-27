import numpy as np
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import argparse


def clean_string(doc):
    tokenizer = WordPunctTokenizer()
    stemmer = ISRIStemmer()
    cur_stopwords = set(stopwords.words('arabic'))
    symbols = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|؟}~\"')
    doc_tokens = tokenizer.tokenize(doc)
    cleaned_tokens = []
    for token in doc_tokens:
        if token in cur_stopwords or token in symbols:
            continue
        cleaned_tokens.append(stemmer.stem(token))
    return " ".join(cleaned_tokens)


def stem_all_docs(docs):
    cleaned_docs = []
    for (i, doc) in enumerate(docs):
        cleaned_docs.append(clean_string(doc))
    return cleaned_docs


class Retriever:
    def __init__(self, top_k_docs, all_docs, ngrams_2, cleaned_docs=None):
        self.top_k_docs = top_k_docs
        self.docs_cpy = all_docs
        if cleaned_docs is None:
            self.docs_stemmed = stem_all_docs(all_docs)
        else:
            self.docs_stemmed = cleaned_docs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, ngrams_2), norm=None, stop_words=stopwords,
                                          lowercase=False)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs_stemmed)

    def get_topk_docs_scores(self, query):
        query = clean_string(query)
        query_tfidf = self.vectorizer.transform([query])
        similarities_raw = linear_kernel(self.tfidf_matrix, query_tfidf)
        similarities = []
        for s in similarities_raw:
            similarities.append(s[0])
        indices_sorted = np.argsort(similarities)[::-1]  # reverse order
        top_docs = []
        docs_scores = []
        for i in range(min(self.top_k_docs, len(self.docs_cpy))):
            top_docs.append(self.docs_cpy[indices_sorted[i]])
            docs_scores.append(similarities[indices_sorted[i]])

        norm_cst = np.sum(np.asarray(docs_scores))
        docs_scores = np.asarray(docs_scores)
        docs_scores = docs_scores / norm_cst
        return top_docs, docs_scores


class HierarchicalRetriever:
    def __init__(self, base_retriever, ngrams_2, top_k_docs_2):
        self.base_retriever = base_retriever
        self.ngrams_2 = ngrams_2
        self.top_k_docs_2 = top_k_docs_2

    def get_topk_docs_scores(self, query):
        docs, _ = self.base_retriever.get_topk_docs_scores(query)
        return Retriever(self.top_k_docs_2, docs, self.ngrams_2).get_topk_docs_scores(query)


def build_tfidf(wiki_path, output_path, ngrams_2, top_k_1, wiki_cleaned):
    wiki_data = pickle.load(open(wiki_path, "rb"))
    stemmed_wiki = pickle.load(open(wiki_cleaned, "rb"))
    r = Retriever(top_k_1, wiki_data, ngrams_2, stemmed_wiki)
    pickle.dump(r, open(output_path, "wb"))


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ngrams", type=int, default=2, help="n-gram order")
parser.add_argument("-k", "--topk", type=int, default=350, help="number of documents retriever should return")
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', default="arwiki_type_para.p")
parser.add_argument('-wc', '--wiki-cleaned', help='Path of arwiki.p', default="arwiki_cleaned_type_para.p")
parser.add_argument('-o', '--output-dir', help='Where to place the retrivers', default="tfidf_stem_retriever.p")

if __name__ == "__main__":
    args = parser.parse_args()
    build_tfidf(args.wiki_path, args.output_dir, args.ngrams, args.topk, args.wiki_cleaned)
