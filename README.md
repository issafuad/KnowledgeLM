# Knowledge based Language Modelling

## Motivation: 
Natural Language understanding (NLU) has advanced significantly in the last decade. One of the achievements that have fueled these advancements is word embeddings. Word embeddings serve to represent words and capture their semantic meaning in NLU tasks. Learning these embeddings is done by training a classifier on a large text corpora to [predict context](https://arxiv.org/pdf/1310.4546.pdf) words given a word and vice-versa. Word2Vec was the first word embeddings trained in such a way and have been shown to capture semantic meaning very well. Further work has produced [FastText]((https://arxiv.org/pdf/1607.04606.pdf)) which were shown to capture more of the syntactic information as they relied on n-gram features within the word. Recently, [Elmo](https://arxiv.org/pdf/1802.05365.pdf) word embeddings were introduced where words are formed by concatenating their constituent character embeddings which are in turn a function of the entire corpus they belong to. Elmo embeddings circumvent the problem of out of vocabulary embeddings by relaying on character embeddings. The most successful word embeddings so far have been the contextualised word embeddings [BERT](https://arxiv.org/pdf/1810.04805.pdf). BERT uses multiple layers of multi-head transformer architecture - mainly multiple self attention layers - over the sentence' words in order to produce contextualised word embeddings. BERT achieves state of the art in many NLU tasks.

Despite all of the previous efforts, none of these approaches tries to explicitly capture semantic relations, factual knowledge or knowledge contained in knowledge bases when learning the representation for these words. Recent investigations have looked into the extent at which language models trained on text contain factual knowledge present in aligned knowledge bases.  The [LAMA probe](https://arxiv.org/pdf/1909.01066.pdf) was introduced to measure how much relational knowledge is present in language models. 

## What to do about this
This project aims to investigate ways to train language models that better capture semantic relations of words and relations present in knowledge bases.

There are two main ideas to achieve that:
 
- Creating a relational multi-document graph from the training corpus and using it to allow the model to have access to further context in other sentences - guided by the graph relations.
- Masking words that belong to the same semantic parse group when training the model. This should encourage contextualised word representations to capture more of the semantic relations between words. 

