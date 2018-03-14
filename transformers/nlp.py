__all__ = ['StanfordCoreNLPTransformer']

import json

import numpy as np

from stanfordcorenlp import StanfordCoreNLP

from ycml.transformers import PureTransformer


class StanfordCoreNLPTransformer(PureTransformer):
    def __init__(
        self,
        corenlp_path='./stanford-corenlp-full-2017-06-09',
        annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'regexner', 'depparse', 'mention'],
        **kwargs
    ):
        kwargs.setdefault('nparray_dtype', np.object)
        super().__init__(**kwargs)

        self.corenlp_path = corenlp_path
        self.nlp_properties = dict(annotators=','.join(annotators), pipelineLanguage='en', outputFormat='json')
        self.nlp = StanfordCoreNLP(self.corenlp_path)
    #end def

    def transform_one(self, instance):
        output = self.nlp.annotate(instance['content'], properties=self.nlp_properties)
        output = json.loads(output)

        instance['corenlp_annotations'] = output

        return instance
    #end def
#end class
