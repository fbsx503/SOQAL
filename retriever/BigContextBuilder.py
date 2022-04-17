import argparse
import json
import os
import pickle
import random
import sys
import time

from CustomRetriever import *
from TfidfRetriever import *


# move every question to a separate paragraph (ziad's version)
def get_big_context_2(retriever, dataset_path, new_dataset_path):
    new_data = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)['data']
    context_total = 0
    context_found = 0

    for article in data:
        new_paragraphs = []
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                new_context = paragraph["context"]
                docs, doc_scores = retriever.get_topk_docs_scores(qa['question'])
                for doc in docs:
                    if paragraph["context"] in doc:
                        context_found += 1
                        new_context = doc
                        break

                context_total += 1

                new_paragraphs.append({
                    "context": new_context,
                    "qas": [qa]
                })
                print("Found context so far: " + str(context_found))
                print("Total Contexts So Far: " + str(context_total))

        new_article = {
            "title": article["title"],
            "paragraphs": new_paragraphs,
        }
        new_data.append(new_article)

    print("Found context " + str(context_found))
    print("Total Contexts " + str(context_total))

    with open(new_dataset_path, 'w') as f:
        json.dump({'data': new_data, 'version': '1.1.1'}, f, indent=4)


def get_big_context(retriever, dataset):
    with open(dataset) as f:
        dataset = json.load(f)['data']
    context_total = 0
    context_found = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    docs, _ = retriever.get_topk_docs_scores(qa['question'])
                    for doc in docs:
                        if doc.find(paragraph['context']) != -1:
                            paragraph['context'] = doc
                            context_found += 1
                    context_total += 1
        print("Found context so far: " + str(context_found))
        print("Total Contexts So Far: " + str(context_total))
    print("Found context " + str(context_found))
    print("Total Contexts " + str(context_total))
    json_string = json.dumps({'data' : dataset})
    with open("bigContext.json", "w") as f1:
        f1.write(json_string)


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=True)
parser.add_argument('-w', '--wiki-path', help='Retriever Path', required=True)
parser.add_argument('-md', '--merged', help='merge docs', required=False, default='f')
parser.add_argument('-data', '--dataset-name', help='dataset-name', required=True)


def main():
    args = parser.parse_args()
    print("Loading data ...")
    base_r = pickle.load(open(args.ret_path, "rb"))
    wiki_data = pickle.load(open(args.wiki_path, "rb"))
    print("Building retriever ...")
    ret = CustomRetriever(base_r, wiki_data, 50, 10, args.merged)
    dataset_path = "../data/" + args.dataset_name
    print("Building big context ...")
    get_big_context(ret, dataset_path)


if __name__ == "__main__":
    main()
