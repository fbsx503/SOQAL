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


def clean_wiki(wiki_path):
    wiki_data = pickle.load(open(wiki_path, "rb"))
    docs_1 = []
    docs_2 = []
    for art, pars in wiki_data.items():
        for par in pars:
            docs_1.append(par)
        docs_2.append(" ".join(pars))
    pickle.dump(stem_all_docs(docs_1), open("arwiki_cleaned_paragraphs.p", "wb"))
    pickle.dump(stem_all_docs(docs_2), open("arwiki_cleaned_articles.p", "wb"))


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', default="arwiki.p")

if __name__ == "__main__":
    args = parser.parse_args()
    clean_wiki(args.wiki_path)
