__all__ = ['generate_paired_mentions_instances', 'count_tokens', 'iter_annotations','find_token_span']

from itertools import combinations
from itertools import permutations


def generate_paired_mentions_instances(nlp_instances, use_gold_mentions=True, ordered_pairs=True, gold_relations_only=False):
    if ordered_pairs:
        def combination_func(L): return permutations(L, 2)
    else:
        def combination_func(L): return combinations(L, 2)

    def _find_relation(relations, m1_id, m2_id):
        for r in relations:
            if r['arg1'] == m1_id and r['arg2'] == m2_id:
                return r
            elif not ordered_pairs and r['arg2'] == m1_id and r['arg1'] == m2_id:
                return r
        #end for

        return False
    #end def

    for nlp_instance in nlp_instances:
        if use_gold_mentions:
            for m1, m2 in combination_func(nlp_instance['mentions']):
                relation = _find_relation(nlp_instance['relations'], m1['id'], m2['id'])
                if gold_relations_only and relation is False: continue

                nlp_instance['paired_mentions'] = dict(m1=m1, m2=m2, relation=relation)

                yield nlp_instance
            #end for
        # else:
        #     corenlp_annotations = instance['corenlp_annotations']
        #     print(json.dumps(corenlp_annotations['sentences'], indent=4, sort_keys=True))
        #end if
    #end for
#end def


def count_tokens(annotations, start, end, inclusive=True):
    return sum(1 for _ in iter_annotations(annotations, start, end, inclusive=inclusive))
#end def


def iter_annotations(annotations, start, end, key='tokens', inclusive=True):
    if start > end:
        start, end = end, start
    #end if

    sentences = annotations['sentences']
    for i in range(start[0], end[0] + 1):
        token_start = 0
        if i == start[0]:  # at the initial sentence
            token_start = start[1] if inclusive else start[1] + 1

        token_end = len(sentences[i])
        if i == end[0]:  # at the end sentence
            token_end = end[1] + 1 if inclusive else end[1]

        for j in range(token_start, token_end):
            yield sentences[i][key][j]
    #end for
#end def


def find_token_span(annotations, m_dict):
    start_char, end_char = m_dict['start_char'], m_dict['end_char']
    span_start = None
    span_end = None
    for i, sent in enumerate(annotations['sentences']):
        for j, token in enumerate(sent['tokens']):
            if start_char >= token['characterOffsetBegin'] and start_char < token['characterOffsetEnd']:
                span_start = (i, j)
            if end_char >= token['characterOffsetBegin'] and end_char < token['characterOffsetEnd']:
                span_end = (i, j)
        #end for
    #end for

    assert span_start is not None and span_end is not None

    return (span_start, span_end)
#end def
