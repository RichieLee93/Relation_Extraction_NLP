from argparse import ArgumentParser, FileType
import json
import logging
import numpy as np
from ycml.transformers import PureTransformer
from .transformers.nlp import StanfordCoreNLPTransformer
from .transformers.lexical import LexicalFeaturesTransformer
from .transformers.dependency import DependencyFeaturesTransformer
from .transformers.feature import FeaturesTransformer
from .utils import generate_paired_mentions_instances

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
logger = logging.getLogger(__name__)


def merge_func(features):
    f1 = features[0]
    f2 = features[1]
    for k in f1:
        yield 'm1'+k
    for k in f2:
        yield 'm2'+k


def main():
    parser = ArgumentParser(description='')
    parser.add_argument('instances', type=FileType('r'), metavar='<instances>', help='Relation extraction instances.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    instances = [json.loads(line) for line in A.instances][:1]
    relation_type = [instance['relations'][0]['type'] for instance in instances]
    le = preprocessing.LabelEncoder()
    le.fit(['Cause-Effect','Instrument-Agency','Product-Producer','Product-Producer','Content-Container',
            'Entity-Origin','Entity-Destination','Component-Whole','Member-Collection','Message-Topic','Other'])
    label = le.transform(relation_type)
    # np.save('data/label.npy', label)
    max_sen_lengh = max(len(i['content'].split()) for i in instances)

    nlp_instances = StanfordCoreNLPTransformer(corenlp_path='./stanford-corenlp-full-2017-06-09').fit_transform(instances)
    relation_instances = list(generate_paired_mentions_instances(nlp_instances, use_gold_mentions=True, ordered_pairs=True))
    p = make_pipeline(FeaturesTransformer())
    p_we_vec = p.fit_transform(relation_instances)
    data = p_we_vec[::2]
    print(data.shape)
    np.save('data/data.npy',data)

#end def


if __name__ == '__main__': main()
