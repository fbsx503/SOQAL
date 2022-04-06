import os
from soqal import SOQAL
import sys
import pickle
from retriever.TfidfRetriever import *
sys.path.append(os.path.abspath("bert"))
from bert.evaluate import *
from araElectra.araelectra import araelectra
from retriever.GoogleSearchRetriever import *
from electra_full_json_builder import *

def accuracy_full_system(AI,dataset):
    with open(dataset) as f:
        dataset = json.load(f)['data']
    exact_match = 0
    f1 = 0
    answers = []
    ground_truths = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    ground_truths.append(answer['text'])
    new_dataset = build_electra_json_from_dataset(dataset)
    total_result = AI.get_answer(new_dataset)
    for result in total_result:
        answers.append(result[0]['text'])

    for i in range(len(answers)):
        exact_match += exact_match_score(ground_truths[i], answers[i])
        f1 += f1_score(ground_truths[i], answers[i])
    print("exact match score 1 = " + str(exact_match/len(answers)))
    print("f1 score 1 = " + str(f1/len(answers)))




def accuracy_system(AI):
    dataset_path = "data/arcd-test-big-context-bertv02.json"
    #dataset_path = "data/arcd.json"
    accuracy_full_system(AI, dataset_path)



def main():
    red = araelectra()
    accuracy_system(red)


if __name__ == "__main__":
    main()
