from rank_bm25 import BM25L
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import pickle
import argparse

tokenizer = WordPunctTokenizer()
stemmer = ISRIStemmer()
stopwords = set(stopwords.words('arabic'))
SYMBOLS = set('!"#$%&\'()؟*+,-./:;<=>?@[\\]^_`{|}~\"')


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


class BM25:
    def __init__(self, top_k_docs, all_docs, cleaned_docs=None):
        self.top_k_docs = top_k_docs
        self.docs_cpy = all_docs
        if cleaned_docs is None:
            self.docs_stemmed = stem_all_docs(all_docs)
        else:
            self.docs_stemmed = cleaned_docs
        self.bm25 = BM25L([doc.split(" ") for doc in self.docs_stemmed])

    def get_topk_docs_scores(self, query):
        query = clean_string(query)
        current_scores = self.bm25.get_scores(query.split(" "))
        best_docs = [x for x, y in sorted(enumerate(current_scores), key=lambda x: x[1])][::-1]
        top_docs = []
        docs_scores = []
        for i in range(min(self.top_k_docs, len(best_docs))):
            top_docs.append(self.docs_cpy[best_docs[i]])
            docs_scores.append(current_scores[best_docs[i]])
        return top_docs, docs_scores


def build_BM25(wiki_path, wiki_path_stem, output_path, top_k_1):
    wiki = pickle.load(open(wiki_path, "rb"))
    stemmed_wiki = pickle.load(open(wiki_path_stem, "rb"))
    r = BM25(top_k_1, wiki, stemmed_wiki)
    pickle.dump(r, open(output_path, "wb"))


parser = argparse.ArgumentParser()
parser.add_argument("-k", "--topk", type=int, default=1000, help="number of documents retriever should return")
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', default="arwiki_paragraphs.p")
parser.add_argument('-ws', '--wiki-path-stem', help='Path of arwiki.p', default="arwiki_cleaned_paragraphs.p")
parser.add_argument('-o', '--output-dir', help='Where to place the retrivers', default="BM25_stem_retriever.p")

if __name__ == "__main__":
    args = parser.parse_args()
    build_BM25(args.wiki_path, args.wiki_path_stem, args.output_dir, args.topk)
