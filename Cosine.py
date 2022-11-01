from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk
nltk.download('stopwords')
import re
import os
import sys
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import transformers

#definition between each 2 sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return (1-cosine_distance(vector1, vector2))


#building similarity matrix 
def build_similarity_matrix(sentences, stop_words):
    
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix
def convertsentences(lis):
    st=""
    for i in lis:
      st+=i
    lis= re.compile('[.!?] ').split(st)
    sentences = []
    for sentence in lis:
      #print(sentence)
      sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    return sentences

#setup rank and pick top ranked sentences
def generate_summary(sentences):
    top_n=2
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences=convertsentences(sentences)
    top_n = len(sentences)//2#minimize size as half
		
    if len(sentences)>0:
      sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
      sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
      scores = nx.pagerank(sentence_similarity_graph)

      ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
      #print("Indexes of top ranked_sentence order are ", ranked_sentence)    
		
      for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
      result=". ".join(summarize_text)
      return result
    else:
      sentences=". ".join(sentences)
      return sentences

#replace your original paragraph in this result string
result = generate_summary("""
A ship is a large watercraft that travels the world's oceans and other sufficiently deep waterways, carrying cargo or passengers, or in support of specialized missions, such as defense, research, and fishing. Ships are generally distinguished from boats, based on size, shape, load capacity, and purpose. Ships have supported exploration, trade, warfare, migration, colonization, and science. After the 15th century, new crops that had come from and to the Americas via the European seafarers significantly contributed to world population growth.[1] Ship transport is responsible for the largest portion of world commerce.

The word ship has meant, depending on the era and the context, either just a large vessel or specifically a ship-rigged sailing ship with three or more masts, each of which is square-rigged.

As of 2016, there were more than 49,000 merchant ships, totaling almost 1.8 billion dead weight tons. Of these 28% were oil tankers, 43% were bulk carriers, and 13% were container ships.[2]
""")
print("\n\nYour cosine similarity based summary is : \n",result)