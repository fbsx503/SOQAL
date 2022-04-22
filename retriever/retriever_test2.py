import json
import random
import pickle
from TfIdfRetriever2 import TfidfRetriever, HierarchicalTfidf
import sys, os

sys.path.append(os.path.abspath("../embedding"))
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=True)


def accuracy_retriever(retriever, dataset):
    with open(dataset) as f:
        dataset = json.load(f)['data']
    found_answers = 0
    total_answers = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    docs, _ = retriever.get_topk_docs_scores(qa['question'])
                    for doc in docs:
                        if doc.find(answer['text']) != -1:
                            found_answers += 1
                            break
                    total_answers += 1
        print("Found answers so far: " + str(found_answers))
        print("Total answers so far: " + str(total_answers))
    print("####################################################")
    print("DONE")
    print("####################################################")
    print("Found answers: " + str(found_answers))
    print("Accuracy is: " + str(found_answers / total_answers))
    print("\n\n")
    return found_answers, total_answers


def accuracy_TfidfRetriever(ret_path):
    r = pickle.load(open(ret_path, "rb"))
    dataset_path = "../data/arcd.json"
    accuracy_retriever(r, dataset_path)


def accuracy_Hierarchial_TfidfRetriever(ret_path, docs, ngrams_1, ngrams_2):
    print("Evaluating Hierarchial TF-IDF Retriever ... {} DOCS, {} ngrams_1, {} ngrams_2".format(docs, ngrams_1,
                                                                                                 ngrams_2))
    r = pickle.load(open(ret_path, "rb"))
    dataset_path = "../data/arcd.json"
    accuracy_retriever(HierarchicalTfidf(r, ngrams_2, docs), dataset_path)


def main():
    args = parser.parse_args()
    print("Evaluating TF-IDF Retriever ...")
    accuracy_TfidfRetriever(args.ret_path)
    accuracy_Hierarchial_TfidfRetriever(args.ret_path, 15, 1, 6)
    accuracy_Hierarchial_TfidfRetriever(args.ret_path, 15, 1, 5)
    accuracy_Hierarchial_TfidfRetriever(args.ret_path, 15, 1, 4)
    accuracy_Hierarchial_TfidfRetriever(args.ret_path, 15, 2, 4)


if __name__ == "__main__":
    main()
