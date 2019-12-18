from preprocessing.graph_creator import Concepts

class WordPieceTokenizer:
    def __init__(self):
        pass

    def sentence2wordpiece(self,
                           sentence: str) -> tuple[list, dict]:
        return word_id_list, index2char

class ConceptTokenizer:
    def __init__(self,
                 concepts: Concepts):
        self.concepts = concepts

    def sentence2concepts(self,
                          sentence: str) -> tuple[list, dict]:
        return concept_id_list, index2char

class InputProducer:
    def __init__(self):
        self.word_tokenizer = WordPieceTokenizer()
        self.concept_tokenizer = ConceptTokenizer()

    def produce_inputs(self,
                       sentence: str):
        word_id_list, index2char = self.word_tokenizer.sentence2wordpiece(sentence)
        concept_id_list, index2char = self.concept_tokenizer.sentence2concepts(sentence)

        #map two mappings to produce
        aligned_concept_id_list = None #complete
        return word_id_list, aligned_concept_id_list