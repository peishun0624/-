# -*- encoding: utf-8 -*-

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing

wiki = WikiCorpus("jawiki-latest-pages-articles.xml.bz2")

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])

document = TaggedWikiDocument(wiki)

model = Doc2Vec(documents=document, dm=1, vector_size=400, window=8, min_count=10, epochs=10, workers=6)
model.save('model/wikipedia.model')

