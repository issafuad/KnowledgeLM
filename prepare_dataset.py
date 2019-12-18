import os
import pickle

from preprocessing.dataset_loader import read_trex, iterate_sentences_from_trex
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)

TREX_PATH = 'tests/test_files/re-nlg_0-10000.json'
TREX_FOLDER = 'TREX'
PROCESSING_OUTPUT_FOLDER = 'dataset_processing'

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# @profile
def prepare_dataset():
    file_list = sorted(os.listdir(TREX_FOLDER))
    for file_number, each_file_name in enumerate(file_list):
        LOGGER.info('Processing file {}/{}'.format(file_number, len(file_list)))
        outfile_name = '{}.{}'.format(os.path.splitext(each_file_name)[0], 'pkl')
        outfile_path = os.path.join(PROCESSING_OUTPUT_FOLDER, outfile_name)

        each_file_path = os.path.join(TREX_FOLDER, each_file_name)
        with open(outfile_path, 'wb') as outfile:
            sentence_list = list(iterate_sentences_from_trex(read_trex(each_file_path)))
            pickle.dump(sentence_list, outfile)

if __name__ == '__main__':
    prepare_dataset()