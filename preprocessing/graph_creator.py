import spacy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from collections import defaultdict
import logging

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

LOGGER = logging.getLogger(__name__)
NLP = spacy.load("en_core_web_sm")


class SentenceMention:
    def __init__(self,
                 id: int,
                 name: str,
                 doc,
                 wiki_id=None):
        self.id = id
        self.name = name

        self.doc = doc
        self.wiki_id = wiki_id
        self.global_id = None


class Sentence:
    #@profile
    def __init__(self,
                 id: int,
                 sentence_text: str,
                 sentence_entities: dict,
                 wiki2surfaceform: dict,
                 doc_info: dict):
        self.id = id
        self.sentence_text = sentence_text
        self.sentence_entities = sentence_entities
        self.wiki2surfaceform = wiki2surfaceform
        self.doc_info = doc_info

        self.graph = None
        self.adjacency_matrix = np.array
        self.global_id = None

    #@profile
    def create_graph(self):
        """
        :param text:
        :param linked_entities: {(4, 27): {'boundaries': [4, 27],
                                            'surfaceform': 'Austroasiatic languages',
                                            'uri': 'http://www.wikidata.org/entity/Q33199',
                                            'annotator': 'Wikidata_Spotlight_Entity_Linker'},
                                (71, 80): {'boundaries': [71, 80],
                                            'surfaceform': 'Mon–Khmer',
                                            'uri': 'http://www.wikidata.org/entity/Q33199',
                                            'annotator': 'Wikidata_Spotlight_Entity_Linker'}, ...
        :return:
        """
        g = nx.DiGraph()

        # Goes through the graph and creates edges and nodes and adds their attributes

        parse = NLP(self.sentence_text)
        node_att_dict = defaultdict(lambda: defaultdict(list))
        for token in parse:
            token_head_range = (token.head.idx, token.head.idx + len(token.head.text))
            token_range = (token.idx, token.idx + len(token.text))
            node_head_name = '{}***{}'.format(token.head.text, token_head_range)
            node_name = '{}***{}'.format(token.text, token_range)

            edge = (node_head_name, node_name, {'dep': token.dep_, 'pos': token.pos_})
            g.add_edge(edge[0], edge[1], attr_dict=edge[2])
            node_att_dict[node_name]['range'].append(token_range)
            node_att_dict[node_name]['name'] = token.text
            # TODO could remove punctuations

        nx.set_node_attributes(g, node_att_dict)
        nx.set_node_attributes(g, self, 'sentence')
        self.graph = g
        if self.wiki2surfaceform:
            self._combine_nodes_in_linked_entities()
        self._string_match_labeling()

    #@profile
    def _combine_nodes_in_linked_entities(self):
        wiki_entities = list()

        for entity_range in self.sentence_entities.keys():

            nodes_to_merge = list()
            for node in self.graph.nodes(data=True):
                if any([each[0] >= entity_range[0] and each[1] <= entity_range[1] for each in node[1]['range']]):
                    nodes_to_merge.append(node)

            wiki_entities.append((self.sentence_entities[entity_range]['uri'], nodes_to_merge))

        assert (len(wiki_entities) == len(self.sentence_entities))

        for (wiki_id, nodes_to_merge_list) in wiki_entities:
            if len(nodes_to_merge_list) > 1:
                for index in range(1, len(nodes_to_merge_list)):
                    self.graph = nx.contracted_nodes(self.graph, nodes_to_merge_list[index][0], nodes_to_merge_list[index - 1][0],
                                            self_loops=False)

                self.graph = nx.relabel.relabel_nodes(self.graph, {nodes_to_merge_list[index][0]: wiki_id})
            elif len(nodes_to_merge_list) == 1:
                self.graph = nx.relabel.relabel_nodes(self.graph, {nodes_to_merge_list[0][0]: wiki_id})

        wiki_attribute = {key: {'surfaceform': self.wiki2surfaceform[key]} for key in self.graph.nodes}
        nx.set_node_attributes(self.graph, wiki_attribute)

    #@profile
    def _string_match_labeling(self):
        relabeling_dict = {each[0]: each[1]['name'] for each in list(self.graph.nodes(data=True)) if
                           each[1].get('surfaceform') is None}
        self.graph = nx.relabel.relabel_nodes(self.graph, relabeling_dict)

    #@profile
    def show_graph(self):
        plt.figure(1, figsize=(100, 100))
        pos = graphviz_layout(self.graph, prog='dot')
        edge_labels = nx.get_edge_attributes(self.graph, 'state')
        nx.draw_networkx_edge_labels(self.graph, pos, labels=edge_labels, node_size=20000, font_size=5)
        nx.draw(self.graph, pos, with_labels=True, arrows=True, node_size=2000, font_size=10)

    def _create_adjacency_matrix(self):
        pass

    def get_nodes_list(self):
        return self.graph.nodes(data=True)

    def get_wiki_nodes(self):
        return [node_name for node_name, attribute_dict in self.graph.nodes(data=True) if attribute_dict.get('surfaceform') is not None]


class Mentions:
    def __init__(self,
                 sentence_list: [Sentence]):

        self.sentence_list = sentence_list
        self.all_graphs = list()
        self.id2mention = dict()
        self._create_mentions_graph()
        # self.mentions_list = self._get_mentions_list(sentence_list)
        self.multidoc_adjacency_matrix = self._get_multidoc_adjacency_matrix(sentence_list)
        assert self.multidoc_adjacency_matrix.shape[0] == len(self.mentions_list)

    def _create_mentions_graph(self):
        current_graph_id = 0
        for sentence in self.sentence_list:
            id2mention = {node_id + current_graph_id: node for node_id, node in enumerate(list(sentence.graph.nodes()))}
            mention2id = {v: k for k, v in id2mention.items()}
            sentence.graph = nx.relabel.relabel_nodes(sentence.graph, mention2id)
            current_graph_id = max(id2mention.keys())
            self.all_graphs.append(sentence.graph)
            self.id2mention.update(id2mention)

    def _get_multidoc_adjacency_matrix(self, doc_list):
        matrix_size = len(self.mentions_list)
        multi_doc_matrix = np.zeros((matrix_size, matrix_size))

        current_matrix_row = 0
        for doc in doc_list:
            doc_matrix = doc.adjacency_matrix
            multi_doc_matrix[current_matrix_row:doc_matrix.shape[0], current_matrix_row:doc_matrix.shape[0]] = doc_matrix
            current_matrix_row += doc_matrix.shape[0]

        return multi_doc_matrix

class Concepts:
    def __init__(self, mentions):
        self.mentions = mentions
        self.concept2mention, surface_form2concept = self._get_concept2mention()
        self.num_concepts = len(self.concept2mention)
        self.id2concept = self._get_id2concept()

    def _get_concept2mention(self):
        concept2mention = defaultdict(list)
        surface_form2concept = defaultdict(list)

        for mention in self.mentions.id2mention.values():
            if mention.wiki_id:
                concept2mention[mention.wiki_id].append(mention)
                surface_form2concept[mention.name].append(mention.wiki_id)
                continue

            concept2mention[mention.name].append(mention)

        return concept2mention, surface_form2concept

    def _get_id2concept(self):
        concept_start_id = self.mentions.number_of_mentions - 1

        id2concept = dict()
        for index, concept in enumerate(self.concept2mention.keys()):
            id2concept[index + concept_start_id] = concept

        return id2concept

    def get_multidoc_adjacency_matrix(self):
        concept_matrix = np.zeros((self.num_concepts, self.mentions.multidoc_adjacency_matrix.shape[1]))
        multidoc_adjacency_matrix = np.concatenate((self.mentions.multidoc_adjacency_matrix, concept_matrix), axis=1)

        for id, concept in self.id2concept.items():
            connected_mentions_global_ids_list = [each.global_id for each in self.concept2mention[concept]]
            multidoc_adjacency_matrix[id, connected_mentions_global_ids_list] = 1

        assert len(self.mentions.id2mention) + len(self.id2concept) == multidoc_adjacency_matrix.shape[1]
        return multidoc_adjacency_matrix


class MultiDocGraph:
    def __init__(self,
                 sentence_list: [Sentence]):
        self.sentence_list = sentence_list
        self.mentions = Mentions(sentence_list)
        self.concepts = Concepts(self.mentions)
        self.multidoc_adjacency_matrix = self.concepts.get_multidoc_adjacency_matrix()

