# Relation_Extraction_NLP
CNN model for relation extraction using rich features
Relation Extraction

## Dev environment setup

Python 3 is the main language used in this codebase.
We strongly encourage the use of Python [virtual environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/):

    virtualenv venv -p /usr/bin/python3
    source venv/bin/activate

After which, you can install the required Python modules via

    pip install -r requirements.txt

You will also need [Stanford Core NLP libraries](https://stanfordnlp.github.io/CoreNLP/download.html).
Download them into your local directory and unzip it:

    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
    unzip stanford-corenlp-full-2017-06-09.zip

## Data Preprocessing

First, we convert the public datasets into our JSON format for ease of use.

    python -m i2r.relation_extraction.jsonify --format ace2005 -i LDC2006T06/ace_2005_td_v7/data/English -o data/ace2005.json

    python -m i2r.relation_extraction.jsonify --format semeval2010 -i SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT -o data/semeval2010.train.json
    python -m i2r.relation_extraction.jsonify --format semeval2010 -i SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT -o data/semeval2010.test.json

We currently have the following formats:

- `semeval2010`: [SemEval 2010 Task 8](http://semeval2.fbk.eu/semeval2.php?location=tasks#T11) dataset.
- `ace2005`: [ACE 2005 Multilingual Training Corpus](https://catalog.ldc.upenn.edu/ldc2006t06) dataset.

An example of JSONified `semeval2010` instance:
```json
{
   "docid": 1,
   "mentions": [
      {
         "id": 1,
         "start_char": 73,
         "end_char": 85,
         "text": "configuration"
      },
      {
         "id": 2,
         "start_char": 98,
         "end_char": 105,
         "text": "elements"
      }
   ],
   "relations": [
      {
         "id": 1,
         "type": "Component-Whole",
         "arg1": 2,
         "arg2": 1
      }
   ],
   "content": "The system as described above has its greatest application in an arrayed configuration of antenna elements.",
   "metadata": {
      "comment": "Comment: Not a collection: there is structure here, organisation."
   },
   "source": "semeval2010"
}
```

## Training a model

Use the `i2r.relation_extraction.train` script to train a relation extraction model.

    python -m i2r.relation_extraction.train data/semeval2010.train.json

### Model Feature Description

#### Lexical Features `LexicalFeaturesTransformer`

- `m1_before_lemma`:
- `m1_2_before_lemma`:
- `m1_after_lemma`:
- `m1_2_after_lemma`:
- `m2_before_lemma`:
- `m2_2_before_lemma`:
- `m2_after_lemma`:
- `mention_token_distance`:
- `mention_char_distance`:
- `m1_before_m2`:
- `m1_token_count`:
- `m2_token_count`:
- `m1_char_count`:
- `m2_char_count`:
- `punctuation_marks_in_between_mentions`:
#### POS Tagging Features `POSFeaturesTransformer`

- `pos_tag_of_m1`:
- `pos_tag_of_m2`:
- `m1_before_pos`:
- `m1_2_before_pos`:
- `m1_after_pos`:
- `m1_2_after_pos`:
- `m2_before_pos`:
- `m2_2_before_pos`:
- `m2_after_pos`:
- `m2_2_after_pos`:

#### Dependency Parsing Features `DependencyFeaturesTransformer`

- `m1_parent_nodes`:
- `m2_parent_nodes`:
- `common_parent`:
- `m1_to_common_parent`:
- `m2_to_common_parent`:
- `dependency_head_path`:
- `dependency_child_path`:
- `dependency_relation_path`:
- `relative_distance1`:
- `relative_distance2`:

#### Word Embedding Features `WordembeddingFeaturesTransformer`

- `word_embedding_of_m1`:
- `word_embedding_of_m2`:
- `avg_word_embedding_in_between`:
- `avg_word_embedding_in_front`:
- `avg_word_embedding_behind`:


