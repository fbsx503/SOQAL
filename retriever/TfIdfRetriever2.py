import numpy as np
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import argparse

tokenizer = WordPunctTokenizer()
stemmer = ISRIStemmer()
stopwords = set(stopwords.words('arabic'))
SYMBOLS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"')
print(stopwords)
print(SYMBOLS)


def clean_string(doc):
    doc_tokens = tokenizer.tokenize(doc)
    cleaned_tokens = []
    for token in doc_tokens:
        if token in stopwords or token in SYMBOLS:
            continue
        cleaned_tokens.append(stemmer.stem(token))
    return " ".join(cleaned_tokens)


def stem_all_docs(docs):
    cleaned_docs = []
    for (i, doc) in enumerate(docs):
        cleaned_docs.append(clean_string(doc))
        if (i % 40000) == 0:
            print("Finished {:.2f}".format(100.00 * i / len(docs)))
    print("Finished Cleaning")
    return cleaned_docs


class TfidfRetriever:
    def __init__(self, top_k_docs, all_docs, ngrams):
        self.top_k_docs = top_k_docs
        self.docs_cpy = all_docs
        self.docs_stemmed = stem_all_docs(all_docs)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), norm=None, stop_words=stopwords, lowercase=False)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs_stemmed)
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
        for i in range(min(self.top_k_docs, len(self.docs_cpy))):
            top_docs.append(self.docs_cpy[indices_sorted[i]])
            docs_scores.append(similarities[indices_sorted[i]])

        norm_cst = np.sum(np.asarray(docs_scores))
        docs_scores = np.asarray(docs_scores)
        docs_scores = docs_scores / norm_cst
        return top_docs, docs_scores


class HierarchicalTfidf:
    def __init__(self, base_retriever, ngrams_2, top_k_docs_2):
        self.base_retriever = base_retriever
        self.ngrams_2 = ngrams_2
        self.top_k_docs_2 = top_k_docs_2

    def get_topk_docs_scores(self, query):
        docs, _ = self.base_retriever.get_topk_docs_scores(query)
        return TfidfRetriever(self.top_k_docs_2, docs, self.ngrams_2).get_topk_docs_scores(query)


def build_tfidf(wiki_path, output_path, ngrams_1, top_k_1):
    wiki_data = pickle.load(open(wiki_path, "rb"))
    docs = []
    for art, pars in wiki_data.items():
        docs.append(" ".join(pars))
    print("finished building documents")
    r = TfidfRetriever(top_k_1, docs, ngrams_1)
    pickle.dump(r, open(output_path, "wb"))


parser = argparse.ArgumentParser()
parser.add_argument("-n1", "--ngrams_1", type=int, default=2, help="n-gram order")
parser.add_argument("-k1", "--topk_1", type=int, default=50, help="number of documents retriever should return")
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', default="arwiki.p")
parser.add_argument('-o', '--output-dir', help='Where to place the retrivers', default="tfidf_stem_retriever.p")

if __name__ == "__main__":
    args = parser.parse_args()
    build_tfidf(args.wiki_path, args.output_dir, args.ngrams_1, args.topk_1)
