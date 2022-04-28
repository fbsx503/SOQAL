import json
import random
import pickle
from TfidfRetriever import TfidfRetriever, HierarchicalTfidf
import sys,os
from BertRanker import *

def accuracy_retriever(retriever, dataset, bert):
    with open(dataset) as f:
        dataset = json.load(f)['data']
    context_total = 0
    context_found = 0
    context_found_bert = 0
    limiter = 9
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                quest = ''
                question_no = 0
                context = paragraph['context']
                quest += qa['question']
                context_total += 1
                question_no += 1
                docs, _ = retriever.get_topk_docs_scores(quest)
                id = 0
                for doc in docs:
                    if id == limiter:
                        break
                    id += 1
                    if doc.find(context) != -1:
                        context_found += question_no
                        break
                print("Found context so far TF-IDF: " + str(context_found))
                new_docs = []
                for doc in docs:
                    val = bert.rank(quest,doc)
                    if val != 1000000000:
                        new_docs.append((doc, bert.rank(quest, doc)))
                new_docs = sorted(new_docs, key=lambda x: -x[1])
                id = 0
                for doc in new_docs:
                    if id == limiter:
                        break
                    id += 1
                    if doc[0].find(context) != -1:
                        context_found_bert += question_no
                        break
                print("Found context so far For Bert: " + str(context_found_bert))
                print("Total Contexts So Far: " + str(context_total))
    print("Found context TF-IDF " + str(context_found))
    print("Found context so far For Bert: " + str(context_found_bert))
    print("Total Contexts " + str(context_total))



def accuracy_Hierarchical():
    base_r = pickle.load(open("tfidfretriever.p", "rb"))
    dataset_path = "../data/arcd.json"
    r = HierarchicalTfidf(base_r, 50, 50)
    bert = BertRanker()
    accuracy_retriever(r, dataset_path, bert)


def main():
    accuracy_Hierarchical()


if __name__ == "__main__":
    main()
