from argparse import ArgumentParser, FileType
import json
import logging
import os
import re
from lxml import etree

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='')
    parser.add_argument('--format', type=str, metavar='<format>', required=True, help='The input format of the dataset.')
    parser.add_argument('-i', '--input', type=str, nargs='+', metavar='<input>', required=True, help='The input locations of dataset to jsonify.')
    parser.add_argument('-o', '--output', type=FileType('w'), metavar='<output>', required=True, help='Save JSONified files here.')
    A = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s', level=logging.INFO)

    if A.format == 'semeval2010': jsonify_func = jsonify_semeval2010
    elif A.format == 'ace2005': jsonify_func = jsonify_ace2005
    else: parser.error('Unknown dataset format: {}'.format(A.format))

    count = 0
    for data_location in A.input:
        location_count = 0
        for o in jsonify_func(data_location):
            assert 'docid' in o
            assert 'content' in o
            assert 'mentions' in o
            assert 'relations' in o

            o.setdefault('source', A.format)

            A.output.write(json.dumps(o))
            A.output.write('\n')

            location_count += 1
            count += 1
        #end for

        logger.info('{} instances found in <{}>.'.format(location_count, data_location))
    #end for
    logger.info('{} instances saved to <{}>.'.format(count, A.output.name))
#end def


def jsonify_semeval2010(data_location):
    # clean some sentences which have no space between words and mentions, e.g. The<e1>staff</e1>...
    clean_no_space_mentions_regex = re.compile(r'([a-z0-9\-])(\<e[12]\>)', flags=re.I)
    relation_regex = re.compile(r'^([a-z\-]+)\((e\d)\,(e\d)\)$', flags=re.I)
    e1_regex = re.compile(r'\<e1\>(.+?)\<\/e1\>')
    e2_regex = re.compile(r'\<e2\>(.+?)\<\/e2\>')

    with open(data_location, 'r') as f:
        cur_instance = []
        for lineno, line in enumerate(f, start=1):
            cur_instance.append(line.rstrip('\n').split('\t'))

            if lineno % 4 == 0:
                instance = dict(docid=int(cur_instance[0][0]), mentions=[], relations=[])

                content = cur_instance[0][1].strip('"')
                content = clean_no_space_mentions_regex.sub(r'\1 \2', content)

                m1 = e1_regex.search(content)
                m2 = e2_regex.search(content)
                e1_text = m1.group(1)
                e2_text = m2.group(1)
                e1_start_char = m1.start(1) - 4
                e2_start_char = m2.start(1) - 13
                e1_end_char = e1_start_char + len(e1_text) - 1
                e2_end_char = e2_start_char + len(e2_text) - 1

                content = e1_regex.sub(r'\1', content)
                content = e2_regex.sub(r'\1', content)
                instance['content'] = content

                assert content[e1_start_char:e1_end_char + 1] == e1_text
                assert content[e2_start_char:e2_end_char + 1] == e2_text
                instance['mentions'].append(dict(id=1, start_char=e1_start_char, end_char=e1_end_char, text=e1_text))
                instance['mentions'].append(dict(id=2, start_char=e2_start_char, end_char=e2_end_char, text=e2_text))

                relation_type = 'Other'
                if cur_instance[1][0] != 'Other':
                    m = relation_regex.match(cur_instance[1][0])
                    relation_type = m.group(1)
                    arg1 = 1 if m.group(2) == 'e1' else 2
                    arg2 = 1 if m.group(3) == 'e1' else 2
                #end if
                instance['relations'].append(dict(id=1, type=relation_type, arg1=arg1, arg2=arg2))

                #jsonify metadata
                instance['metadata'] = dict(comment=cur_instance[2][0])

                yield instance

                cur_instance = []
            #end if
        #end for
    #end with
#end def


def jsonify_ace2005(data_location):
    source_dir = {}
    for dirname in os.listdir(data_location):
        dirpath = os.path.join(data_location, dirname)
        if os.path.isdir(dirpath):
            source_dir[dirname] = os.path.join(dirpath, 'adj')
    #end for

    def _jsonify_ace2005_instance(docid, base_path):
        instance = dict(docid=docid, mentions=[], relations=[])

        with open(base_path + '.sgm', 'r') as f: sgm_content = f.read()
        # sgm_content = re.sub(r'\<[A-Z]+[.\n]*?\>', '', sgm_content, flags=re.M)
        # sgm_content = re.sub(r'\<\/[A-Z]+\>', '', sgm_content)
        # instance['content'] = sgm_content
        sgm_content = re.sub(r'\&', '\u039d', sgm_content)
        sgm_root = etree.fromstring(sgm_content)
        content = ''.join(sgm_root.itertext())
        content = content.replace('\u039d', '&')
        # sgm_tree = etree.parse(base_path + '.sgm')
        # sgm_root = sgm_tree.getroot()
        instance['content'] = content

        apf_tree = etree.parse(base_path + '.apf.xml')
        apf_root = apf_tree.getroot()
        relation = []
        for relation in apf_root.iterfind('.//relation'):
            relation_type = relation.get('TYPE')
            relation_subtype = relation.get('SUBTYPE')

            for relation_mention in relation.iterfind('./relation_mention'):
                relation_id = relation_mention.get('ID')
                relation_dict = dict(id=relation_id, type=relation_type, subtype=relation_subtype)

                for relation_mention_argument in relation_mention.iterfind('./relation_mention_argument'):
                    mention_id = relation_mention_argument.get('REFID')
                    charseq = relation_mention_argument.find('./extent/charseq')
                    start_char = int(charseq.get('START'))
                    end_char = int(charseq.get('END'))
                    text = re.sub(r'\&([^a])', r'&amp;\1', charseq.text)

                    assert mention_id in ['BACONSREBELLION_20050226.1317-E39-74', 'BACONSREBELLION_20050226.1317-E38-73'] or instance['content'][start_char:end_char + 1] == text

                    mention_dict = dict(id=mention_id, start_char=start_char, end_char=end_char, text=text)
                    entity_mention = apf_root.find('.//entity_mention[@ID="{}"]'.format(mention_id))
                    if entity_mention is not None:
                        mention_dict['type'] = entity_mention.get('TYPE')
                        mention_dict['role'] = entity_mention.get('ROLE')
                        entity = entity_mention.getparent()
                        mention_dict['entity_type'] = entity.get('TYPE')
                        mention_dict['entity_subtype'] = entity.get('SUBTYPE')
                    #end if
                    instance['mentions'].append(mention_dict)
                    # if instance['content'][start_char:end_char + 1] != text:
                    #     print(base_path, mention_id)
                    #     # print(instance['content'])
                    #     print('instance', instance['content'][start_char:end_char + 1])
                    #     print('text', text)
                    # #end if
                    #end if

                    role = relation_mention_argument.get('ROLE')
                    m = re.match(r'^Arg\-(\d+)$', role)
                    if m:
                        i = int(m.group(1))
                        relation_dict['arg{}'.format(i)] = mention_id
                    else:
                        relation_dict[role] = mention_id
                #end for

                instance['relations'].append(relation_dict)
            #end for
        #end for

        return instance
    #end def

    for source_type, dirpath in sorted(source_dir.items()):
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith('.tab'): continue
            docid, _ = os.path.splitext(fname)
            yield _jsonify_ace2005_instance(docid, os.path.join(dirpath, docid))
        #end for
    #end for
#end def


if __name__ == '__main__': main()
