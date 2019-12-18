from preprocessing.graph_creator import Sentence

import json
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Iterable
import logging
from multiprocessing import Pool
from functools import partial

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)

def read_trex(file_path):

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return data


def find_linked_entities(trex_list: List[dict]) -> Dict[str, set]:
    doc_entities = list()
    for current_doc in trex_list:
        doc_entities.extend([each for each in current_doc['entities'] if 'Entity' in each['annotator']])

    wiki2surfaceform = defaultdict(set)
    for entity_dict in doc_entities:
        wiki2surfaceform[entity_dict['uri']].add(entity_dict['surfaceform'])

    return wiki2surfaceform


def remove_overlapping_sentence_entities(sentence_entities):
    def range_in(range_set, range_):
        return any([range_[0] >= each_range[0] and range_[1] <= each_range[1] for each_range in range_set])

    def range_partially_in(range_set, range_):
        return any([(range_[0] >= each_range[0] and range_[0] <= each_range[1]) or
                    (range_[1] >= each_range[0] and range_[1] <= each_range[1]) for each_range in range_set])

    range_set = set(sentence_entities.keys())
    range_set_2 = deepcopy(range_set)

    for range_, sentence_entity_dict in list(sentence_entities.items()):
        exclude_range_from_range_set = {each for each in range_set if each != range_}
        exclude_range_from_range_set_2 = {each for each in range_set_2 if each != range_}
        if range_in(exclude_range_from_range_set, range_):
            del sentence_entities[range_]
        elif range_partially_in(exclude_range_from_range_set_2, range_):
            del sentence_entities[range_]
            range_set_2.remove(range_)


def _get_sentence_text_and_entities(trex_list: List[dict]) -> Iterable[Sentence]:

    for current_doc in trex_list:
        doc_entities = [each for each in current_doc['entities'] if 'Entity' in each['annotator']]
        doc_list = list()
        for sentence_boundary in current_doc['sentences_boundaries']:
            sentence_entities = deepcopy([each for each in doc_entities if
                                          each['boundaries'][1] <= sentence_boundary[1] and each['boundaries'][0] >=
                                          sentence_boundary[0]])
            for each in sentence_entities:
                each['boundaries'] = [each['boundaries'][0] - sentence_boundary[0],
                                      each['boundaries'][1] - sentence_boundary[0]]

            sentence_entities_dict = {tuple(each['boundaries']): each for each in sentence_entities}
            sentence_text = current_doc['text'][sentence_boundary[0]: sentence_boundary[1]]
            doc_list.append((sentence_text, sentence_entities_dict, current_doc))

        yield doc_list

# @profile
def iterate_sentences_from_trex(trex_list: List[dict]) -> Iterable[Sentence]:
    wiki2surfaceform = find_linked_entities(trex_list)
    LOGGER.info('Number of documents in this file : {}'.format(len(trex_list)))

    sentence_id = 0
    pool = Pool(processes=4)
    create_sentence_partial = partial(create_sentence, wiki2surfaceform=wiki2surfaceform)

    for doc_num, doc_list in enumerate(_get_sentence_text_and_entities(trex_list)):
        doc_list_with_sentence_id = list()
        for sentence_number, (sentence_text, sentence_entities, doc_info) in enumerate(doc_list):
            remove_overlapping_sentence_entities(sentence_entities)
            doc_list_with_sentence_id.append((sentence_id, sentence_text, sentence_entities, doc_info))
            LOGGER.info('Processed {} sentences in {}/{} docs'.format(sentence_number + 1, doc_num + 1, len(trex_list)))
            LOGGER.info('Sentence Text : {}'.format(sentence_text))
            sentence_id += 1

        for sentence_number, (sentence_id, sentence_text, sentence_entities, doc_info) in enumerate(doc_list_with_sentence_id):
            LOGGER.info('Processed {} sentences in {}/{} docs'.format(sentence_number + 1, doc_num + 1, len(trex_list)))
            LOGGER.info('Sentence Text : {}'.format(sentence_text))
            yield create_sentence_partial(sentence_id, sentence_text, sentence_entities, doc_info)

        # for sentence in pool.starmap(create_sentence_partial, doc_list_with_sentence_id):
        #     yield sentence

# @profile
def create_sentence(id_counter, sentence_text, sentence_entities, doc_info, wiki2surfaceform=None):
    sentence = Sentence(id=id_counter,
                        sentence_text=sentence_text,
                        sentence_entities=sentence_entities,
                        wiki2surfaceform=wiki2surfaceform,
                        doc_info=doc_info)
    sentence.create_graph()

    return sentence

