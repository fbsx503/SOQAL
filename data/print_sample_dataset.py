import json

with open('ARCD_Squad_preprocessed.json') as json_file:
    data = json.load(json_file)
    print(data['data'][2]['paragraphs'][0]['context'])
