# {"data":[{"paragraphs": [{"qas": [{"question": "aa", "id": "q_0"}], "context": "aa"}]}]}
def build_electra_json_from_dataset(old_dataset):
    paragraphs = []
    id_i = 0
    for article in old_dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                context = paragraph['context']
                qas = []
                id = str(id_i)
                quest = {
                    'question': question,
                    'id': 'q_' + id
                }
                qas.append(quest)
                paragraph = {
                    'qas': qas,
                    'context': context
                }
                paragraphs.append(paragraph)
                id_i += 1
    return {"data": [{"paragraphs": paragraphs}]}