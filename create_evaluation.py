import os
import sys

sys.path.append(os.path.abspath("retriever"))
from retriever.TfidfRetriever import *
from retriever.GoogleSearchRetriever import *
from soqal import SOQAL

sys.path.append(os.path.abspath("bert"))
from bert.Bert_model import BERT_model
from bert.evaluate import *

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=False, default='retriever/tfidfretriever.p')
parser.add_argument('-rc', '--retCache', help='Retriever cache', required=False, default='t')
parser.add_argument('-pm', '--pre-model', help='Preprocess model', required=False, default=None)
parser.add_argument('-a', '--aggregate', help='Aggregate function', required=False, default='o')


def main():
    dataset_path = "./data/tydiqa-goldp-dev-arabic.json"
    with open(dataset_path) as f:
        dataset = json.load(f)['data']
    args = parser.parse_args()
    base_r = pickle.load(open(args.ret_path, "rb"))
    ret = HierarchicalTfidf(base_r, 50, 50)
    AI = SOQAL(ret, None, 0.999, args.pre_model, args.aggregate)
    AI.dump_new_dataset(args, dataset)


if __name__ == "__main__":
    main()
