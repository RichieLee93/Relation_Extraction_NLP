__all__ = ['DependencyFeaturesTransformer']

import numpy as np

from ycml.transformers import PureTransformer

from ..utils import count_tokens
from ..utils import iter_annotations
from ..utils import find_token_span

class DependencyFeaturesTransformer(PureTransformer):
    def __init__(self, **kwargs):
        kwargs.setdefault('nparray_dtype', np.object)
        super().__init__(**kwargs)
    #end def

    def transform_one(self, relation_instance):
        m1_dict, m2_dict = relation_instance['paired_mentions']['m1'], relation_instance['paired_mentions']['m2']

        corenlp_annotations = relation_instance['corenlp_annotations']
        sentences = corenlp_annotations['sentences']

        m1_span = find_token_span(corenlp_annotations, m1_dict)
        m2_span = find_token_span(corenlp_annotations, m2_dict)

        features = {}
        m1_index = sentences[m1_span[1][0]]['tokens'][m1_span[1][1]]['index']
        m2_index = sentences[m2_span[1][0]]['tokens'][m2_span[1][1]]['index']
        m1_index_cp = m1_index_cp2 = m1_index
        m2_index_cp = m2_index_cp2 = m2_index
        m1_parent = []
        m2_parent = []
        m1_parent_index = []
        m2_parent_index = []
        while m1_index != 0:
            for d in sentences[m1_span[1][0]]['enhancedPlusPlusDependencies']:
                if d['dependent'] == m1_index:
                    m1_index = d['governor']
                    m1_parent_index.append(m1_index)
                    m1_parent.append(d['governorGloss'])
        while m2_index != 0:
            for d in sentences[m2_span[1][0]]['enhancedPlusPlusDependencies']:
                if d['dependent'] == m2_index:
                    m2_index = d['governor']
                    m2_parent_index.append(m2_index)
                    m2_parent.append(d['governorGloss'])
        features['m1_parent_nodes'] = str(m1_parent)
        features['m2_parent_nodes'] = str(m2_parent)

        common_parent = []
        common_parent_index = []
        for p1 in m1_parent_index:
            if p1 in m2_parent_index:
                common_parent_index.append(p1)
        for p1 in m1_parent:
            if p1 in m2_parent:
                common_parent.append(p1)
        features['common_parent'] = str(common_parent)

        first_cp_index_m1 = m1_parent.index(common_parent[0])
        m1_to_cp = m1_parent[:first_cp_index_m1+1]
        features['m1_to_common_parent'] = str(m1_to_cp)

        first_cp_index_m2 = m2_parent.index(common_parent[0])
        m2_to_cp = m2_parent[:first_cp_index_m2+1]
        features['m2_to_common_parent'] = str(m2_to_cp)

        dep_head_path = []
        while m1_index_cp != common_parent_index[0]:
            for d in sentences[m1_span[1][0]]['enhancedPlusPlusDependencies']:
                if d['dependent'] == m1_index_cp:
                    m1_index_cp = d['governor']
                    dep_head_path.append(d['governorGloss'])
        while m2_index_cp != common_parent_index[0]:
            for d in sentences[m2_span[1][0]]['enhancedPlusPlusDependencies']:
                if d['dependent'] == m2_index_cp:
                    m2_index_cp = d['governor']
                    dep_head_path.append(d['governorGloss'])
        features['dependency_head_path'] = str(dep_head_path)

        dep_child_path = []
        dep_relation_path = []
        while m1_index_cp2 != common_parent_index[0]:
            for d in sentences[m1_span[1][0]]['enhancedPlusPlusDependencies']:
                if d['dependent'] == m1_index_cp2:
                    dep_child_path.append(d['dependentGloss'])
                    dep_relation_path.append(d['dep'])
                    m1_index_cp2 = d['governor']
        while m2_index_cp2 != common_parent_index[0]:
            for d in sentences[m2_span[1][0]]['enhancedPlusPlusDependencies']:
                if d['dependent'] == m2_index_cp2:
                    dep_child_path.append(d['dependentGloss'])
                    dep_relation_path.append(d['dep'])
                    m2_index_cp2 = d['governor']
        features['dependency_child_path'] = str(dep_child_path)
        features['dependency_relation_path'] = str(dep_relation_path)

        relative_distance1 = len(dep_relation_path)
        features['relative_distance1'] = str(relative_distance1)

        relative_distance2 = []
        l = len(m1_to_cp)
        r = len(m2_to_cp)
        for i in range(0,l):
            for j in range(0,r):
                relative_distance2.append((i,j))
        features['relative_distance2'] = str(relative_distance2)

        return list(features.values())
    #end def
#end class

