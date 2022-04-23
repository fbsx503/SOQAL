import os

import sys
import pickle

sys.path.append(os.path.abspath("retriever"))
from retriever.TfidfRetriever import *
from retriever.GoogleSearchRetriever import *
from retriever.CustomRetriever import *

sys.path.append(os.path.abspath("bert"))
from bert.Bert_model import BERT_model
from bert.evaluate import *
from soqal import SOQAL


def accuracy_full_system(AI, dataset, args):
    with open(dataset) as f:
        dataset = json.load(f)['data']
    AI.ask_all(dataset, args)


def accuracy_system(AI, args):
    dataset_path = "data/arcd-test.json"
    accuracy_full_system(AI, dataset_path, args)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Path to bert_config.json', required=True)
parser.add_argument('-v', '--vocab', help='Path to vocab.txt', required=True)
parser.add_argument('-o', '--output', help='Directory of model outputs', required=True)
parser.add_argument('-g', '--google', help='use tf-idf or google', required=False, default='f')
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=False, default='retriever/tfidfretriever.p')
parser.add_argument('-rc', '--retCache', help='Retriever cache', required=False, default='t')
parser.add_argument('-pm', '--pre-model', help='Preprocess model', required=False, default=None)
parser.add_argument('-w', '--wiki-path', help='Wikipedia Path', required=False, default='f')
parser.add_argument('-md', '--merged', help='merge docs', required=False, default='t')
parser.add_argument('-rpa', '--ret-per-article', help='Retriever documents per question or whole article',
                    required=False, default='article')


def main():
    args = parser.parse_args()
    if args.google == 't':
        base_r = pickle.load(open(args.ret_path, "rb"))
        wiki_data = pickle.load(open(args.wiki_path, "rb"))
        ret = CustomRetriever(base_r, wiki_data, 50, 10, args.merged)
    else:
        base_r = pickle.load(open(args.ret_path, "rb"))
        ret = HierarchicalTfidf(base_r, 50, 50)

    red = BERT_model(args.config, args.vocab, args.output)
    AI = SOQAL(ret, red, 0.999, args.pre_model)
    accuracy_system(AI, args)


if __name__ == "__main__":
    main()
