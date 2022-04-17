from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
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


def clean_wiki(wiki_path, type):
    wiki_data = pickle.load(open(wiki_path, "rb"))
    docs = []
    if type == 'para':
        print("Extracting paragraphs!")
    else:
        print("Extracting articles!")
    for art, pars in wiki_data.items():
        if type == 'para':
            for par in pars:
                docs.append(par)
        else:
            docs.append(" ".join(pars))
    pickle.dump(docs, open("arwiki_type_{}.p".format(type), "wb"))
    pickle.dump(stem_all_docs(docs), open("arwiki_cleaned_type_{}.p".format(type), "wb"))


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wiki-path', help='Path of arwiki.p', default="arwiki.p")
parser.add_argument('-t', '--type', help='Paragraph/Article', default="para")

if __name__ == "__main__":
    args = parser.parse_args()
    clean_wiki(args.wiki_path, args.type)
