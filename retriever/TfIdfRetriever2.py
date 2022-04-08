import numpy as np
from tashaphyne.stemming import ArabicLightStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import argparse

tokenizer = WordPunctTokenizer()
stemmer = ArabicLightStemmer()
stopwords = stopwords.words('arabic')
SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"'


def has_symbol(token):
    for s in SYMBOLS:
        if s in token:
            return True
    return False


def clean_string(doc):
    doc_tokens = tokenizer.tokenize(doc)
    cleaned_tokens = []
    for token in doc_tokens:
        if token in stopwords or has_symbol(token):
            continue
        cleaned_tokens.append(stemmer.light_stem(token))
    return " ".join(cleaned_tokens)


def stem_all_docs(docs):
    cleaned_docs = []
    for (i, doc) in enumerate(docs):
        cleaned_docs.append(clean_string(doc))
        if (i % 40000) == 0:
            print("Finised {:.2f}".format(1.00 * i / len(docs)))
    print("Finished Cleaning")
    return cleaned_docs


class TfidfRetriever:
    def __init__(self, top_k_docs, all_docs, ngrams):
        self.top_k_docs = top_k_docs
        self.docs = all_docs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), norm=None, stop_words=stopwords)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
        print("Finished TFIDF fit-transform")

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
        for i in range(min(self.top_k_docs, len(self.docs))):
            top_docs.append(self.docs[indices_sorted[i]])
            docs_scores.append(similarities[indices_sorted[i]])

        norm_cst = np.sum(np.asarray(docs_scores))
        docs_scores = np.asarray(docs_scores)
        docs_scores = docs_scores / norm_cst
        return top_docs, docs_scores


class HierarchicalTfidf:
    def __init__(self, top_k_docs_retriever_1, top_k_docs_retriever_2, all_docs, ngram_1, ngram_2):
        self.base_retriever = TfidfRetriever(top_k_docs_retriever_1, all_docs, ngram_1)
        self.ngram_2 = ngram_2
        self.top_k_docs_retriever_2 = top_k_docs_retriever_2

    def get_topk_docs_scores(self, query):
        docs, _ = self.base_retriever.get_topk_docs_scores(query)
        return TfidfRetriever(self.top_k_docs_retriever_2, docs, self.ngram_2).get_topk_docs_scores(query)


def build_hierarchical_tfidf(wiki_path, output_path, ngrams_1, ngrams_2, top_k_1, top_k_2):
    wiki_data = pickle.load(open(wiki_path, "rb"))
    docs = []
    for art, pars in wiki_data.items():
        docs.append(" ".join(pars))
    print("finished building documents")
    print("Stemming Docs")
    docs = stem_all_docs(docs)
    print("Finished Stemming Docs")
    r = HierarchicalTfidf(top_k_1, top_k_2, docs, ngrams_1, ngrams_2)
    pickle.dump(r, open(output_path + "/hierarchical_tfidf.p", "wb"))


parser = argparse.ArgumentParser()
parser.add_argument("-n1", "--ngrams_1", type=int, default=2, help="n-gram order")
parser.add_argument("-n2", "--ngrams_2", type=int, default=1, help="n-gram order")
parser.add_argument("-k1", "--topk_1", type=int, default=50, help="number of documents retriever should return")
parser.add_argument("-k2", "--topk_2", type=int, default=10, help="number of documents retriever should return")
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', default="arwiki.p")
parser.add_argument('-o', '--output-dir', help='Where to place the retrivers', default="")

if __name__ == "__main__":
    args = parser.parse_args()
    build_hierarchical_tfidf(args.wiki_path, args.output_dir, args.ngrams_1, args.ngrams_2, args.topk_1, args.topk_2)
