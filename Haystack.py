import json
import logging
from pprint import pprint
import elasticsearch
from tqdm.auto import tqdm
from haystack.nodes import QuestionGenerator, BM25Retriever, FARMReader
from haystack import Document
#from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import JsonConverter
from haystack.pipelines import (
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,
)
from haystack.utils import launch_es, print_questions, convert_files_to_docs

import os
from subprocess import Popen, PIPE, STDOUT

from trio import sleep
from haystack.telemetry import tutorial_running
tutorial_running(10)

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = convert_files_to_docs(dir_path="/uploads")

def get_questions(text):
    question_generator = QuestionGenerator()
    reader = FARMReader("deepset/roberta-base-squad2")
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
    doc_json = [{"content": text}]
    json_string = json.dumps(doc_json)
    document_obj = Document.from_json(doc_json)
    result = qag_pipeline.run(documents=[document_obj])
    return print_questions(result)

#pip install gensim
#pip install jieba
from gensim import corpora, models, similarities
import jieba

def compare_answers(user_ans, correct_answer):
    text = user_ans
    keyword = correct_answer
    dictionary = corpora.Dictionary(text)
    feature_cnt = len(dictionary.token2id)
    corpus = dictionary.doc2bow(text) #gets the dictionary and the words used. 
    tfidf = models.TfidfModel(corpus) 
    kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
    sim = index[tfidf[kw_vector]] #finds the similarity index between the user answer. 
    if sim < 0.75:
        return True
    else:
        return False
