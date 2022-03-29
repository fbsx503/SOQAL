import os
from soqal import SOQAL
import sys
import pickle
sys.path.append(os.path.abspath("retriever"))
from retriever.TfidfRetriever import *
sys.path.append(os.path.abspath("bert"))
from bert.evaluate import *
from araElectra.QA import QA
from araElectra.tf.Araelectra import Araelectra
from retriever.GoogleSearchRetriever import *


def accuracy_full_system(AI, dataset):
    with open(dataset) as f:
        dataset = json.load(f)['data']
    question_no = 0
    exact_match_1 = 0
    exact_match_3 = 0
    exact_match_5 = 0
    f1_1 = 0
    f1_3 = 0
    f1_5 = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                response = AI.ask_araelectra(qa['question'])
                question_no = question_no + 1
                exact_match_max_1 = 0
                f1_max_1 = 0
                f1_max_3 = 0
                exact_match_max_3 = 0
                f1_max_5 = 0
                exact_match_max_5 = 0
                print('Context: ' + str(paragraph['context']))
                for answer in qa['answers']:
                    print("Question number is: " + str(question_no))
                    print("Question is: " + qa['question'])
                    print("result is: " + str(response))
                    print("ground truth is: " + str(answer['text']))
                    for i in range(len(response)):
                        if i < 1:
                            exact_match_max_1 = max(exact_match_max_1, exact_match_score(response[i], answer['text']))
                            f1_max_1 = max(f1_max_1, f1_score(response[i], answer['text']))
                        if i < 3:
                            exact_match_max_3 = max(exact_match_max_3, exact_match_score(response[i], answer['text']))
                            f1_max_3 = max(f1_max_3, f1_score(response[i], answer['text']))
                        if i < 5:
                            exact_match_max_5 = max(exact_match_max_5, exact_match_score(response[i], answer['text']))
                            f1_max_5 = max(f1_max_5, f1_score(response[i], answer['text']))

                exact_match_1 += exact_match_max_1
                f1_1 += f1_max_1
                exact_match_3 += exact_match_max_3
                f1_3 += f1_max_3
                exact_match_5 += exact_match_max_5
                f1_5 += f1_max_5
                print("exact match score 1 = " + str(exact_match_1))
                print("exact match score 3 = " + str(exact_match_3))
                print("exact match score 5 = " + str(exact_match_5))
                print("f1 score 1 = " + str(f1_1))
                print("f3 score 1 = " + str(f1_3))
                print("f5 score 1 = " + str(f1_5))
                print("percent exact match score 1 = " + str(exact_match_1 / question_no))
                print("percent exact match score 3 = " + str(exact_match_3 / question_no))
                print("percent exact match score 5 = " + str(exact_match_5 / question_no))
                print("percent f1 score 1 = " + str(f1_1 / question_no))
                print("percent f3 score 1 = " + str(f1_3 / question_no))
                print("percent f5 score 1 = " + str(f1_5 / question_no))
    print("exact match score 1 = " + str(exact_match_1 / question_no))
    print("exact match score 3 = " + str(exact_match_3 / question_no))
    print("exact match score 5 = " + str(exact_match_5 / question_no))
    print("f1 score 1 = " + str(f1_1 / question_no))
    print("f3 score 1 = " + str(f1_3 / question_no))
    print("f5 score 1 = " + str(f1_5 / question_no))




def accuracy_system(AI):
    dataset_path = "data/arcd-test.json"
    accuracy_full_system(AI, dataset_path)


import  argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--google', help='use tf-idf or google', required=True)
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=True)

def main():
    args = parser.parse_args()
    if args.google == 't':
        doc_number = 10
        wiki_data = pickle.load(open(args.ret_path, "rb"))
        ret = ApiGoogleSearchRetriever(wiki_data, doc_number)
    else:
        base_r = pickle.load(open(args.ret_path, "rb"))
        ret = HierarchicalTfidf(base_r, 50, 50)
    red = Araelectra()
    AI = SOQAL(ret, red, beta)
    accuracy_system(AI)


if __name__ == "__main__":
    main()
