import numpy as np
import sys
import pickle
import json


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SOQAL:
    def __init__(self, retriever, reader, beta):
        self.retriever = retriever
        self.beta = beta
        self.reader = reader

    def build_quest_json(self, quest, docs):
        articles = []
        paragraphs = []
        id_i = 0
        for doc in docs:
            paragraph_context = doc
            qas = []
            id = str(id_i)
            ques = quest
            ans = ""
            answer_start = 0
            answer = {
                'text': ans,
                'answer_start': answer_start
            }
            question = {
                'question': ques,
                'id': id,
                'answers': [answer]
            }
            qas.append(question)
            paragraph = {
                'context': paragraph_context,
                'qas': qas
            }
            paragraphs.append(paragraph)
            id_i += 1
        article = {
            'title': "prediction",
            'paragraphs': paragraphs
        }
        articles.append(article)
        return articles

    def build_quest_json_araElectra(self, docs):
        paragraph = []
        for doc in docs:
            paragraph.append(doc)
        return paragraph

    def get_predictions(self, predictions_raw):
        answers_text = []
        answers_scores = []
        for i in range(0, len(predictions_raw)):
            doc_ques_id = str(i)
            # pick the first as the highest, better to pick all
            for j in range(0, min(1, len(predictions_raw))):
                pred = predictions_raw[doc_ques_id][j]
                pred_score = pred['start_logit'] * pred['end_logit']
                pred_answer = pred['text']
                answers_text.append(pred_answer)
                answers_scores.append(pred_score)
        return answers_text, answers_scores

    def electra_agreggate(self, answers_text, answers_scores, docs_scores):
        pred = []
        ans_indx = np.argsort(answers_scores)[::-1]
        pred.append(answers_text[ans_indx[0]])
        for i in range(3):
            pred.append(answers_text[i])
        pred.append(answers_text[ans_indx[1]])
        return pred

    def bert_agreggate(self, answers_text, answers_scores, docs_scores):
        ans_scores = np.asarray(answers_scores)
        doc_scores = np.asarray(docs_scores)
        final_scores = (1 - self.beta) * softmax(ans_scores) + self.beta * softmax(doc_scores)
        ans_indx = np.argsort(final_scores)[::-1]
        pred = []
        for k in range(0, min(5, len(ans_indx))):
            pred.append(answers_text[ans_indx[k]])
        print("aggregated answers here")
        print(pred)
        return pred

    def ask(self, quest):
        docs, doc_scores = self.retriever.get_topk_docs_scores(quest)
        print("got documents")
        dataset = self.build_quest_json(quest, docs)
        print("built documents json")
        nbest = self.reader.predict_batch(dataset)
        print("got predictions from BERT")
        answers, answers_scores = self.get_predictions(nbest)
        prediction = self.bert_agreggate(answers, answers_scores, doc_scores)
        return prediction

    def ask_araelectra1(self, quest):
        docs, doc_scores = self.retriever.get_topk_docs_scores(quest)
        print("got documents")
        dataset = self.build_quest_json_araElectra(docs)
        print("built documents json")
        total_result = []
        for context in dataset:
            context = self.reader.preprocess(context)
            if len(context) < 2:
                continue
            total_result.append(self.reader.answerQuestion(question=quest, context=context))
        result = sorted(total_result, key=lambda object1: object1["score"], reverse=True)
        answers = []
        for i in range(0, 5):
            answers.append(result[i]["answer"])
        return answers

    def ask_araelectra(self, quest):
        docs, doc_scores = self.retriever.get_topk_docs_scores(quest)
        print("got documents")
        dataset = self.build_quest_json_araElectra(docs)
        print("built documents json")
        total_result = []
        id = 0
        for context in dataset:
            if len(context) < 2:
               doc_scores = np.delete(doc_scores, id)
               continue
            id += 1
            total_result.append(self.reader.get_answer(quest, context))
        answers = []
        answer_scores = []
        for result in total_result:
            answers.append(result['text'])
            answer_scores.append(result['start_logit'] * result['end_logit'])
        prediction = self.electra_agreggate(answers, answer_scores, doc_scores)
        return prediction
