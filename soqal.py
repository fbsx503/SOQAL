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

    def agreggate(self, answers_text, answers_scores, docs_scores):
        pred = []
        ans_indx = np.argsort(answers_scores)[::-1]
        pred.append(answer_text[ans_indx[0]])
        for i in range(3):
            pred.append(answer_text[i])
        pred.append(answer_text[ans_indx[1]])
        return pred

    def ask(self, quest):
        docs, doc_scores = self.retriever.get_topk_docs_scores(quest)
        print("got documents")
        dataset = self.build_quest_json(quest, docs)
        print("built documents json")
        nbest = self.reader.predict_batch(dataset)
        print("got predictions from BERT")
        answers, answers_scores = self.get_predictions(nbest)
        prediction = self.agreggate(answers, answers_scores, doc_scores)
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
        dataset = self.build_quest_json_araElectra(docs)
        total_result = []
        id = 0
        for context in dataset:
            context = self.reader.preprocess(context)
            #print("context len"+str(len(context)))
            if len(context) < 2:
               doc_scores = np.delete(doc_scores, id)
               continue
            id += 1
            import textwrap
            lines = textwrap.wrap(context,350, break_long_words=False)
            best_result=(self.reader.answerQuestion(question=quest, context=context))
            for line in lines:
                result=(self.reader.answerQuestion(question=quest, context=line))
                if best_result['score'] < result['score']:
                   best_result=result
            total_result.append(best_result)
        answers = []
        answer_scores = []
        for result in total_result:
            answers.append(self.reader.unpreprocess(result['answer']))
            answer_scores.append(result['score'])
        prediction = self.agreggate(answers, answer_scores, doc_scores)
        return prediction
