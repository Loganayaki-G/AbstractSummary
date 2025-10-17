#pip install transformers
from transformers import pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization")#To use the BART model, which is trained on the CNN/Daily Mail News Dataset, 
#or
#summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")#If you want to use the t5 model (e.g. t5-base), which is trained on the c4 Common Crawl web corpus, then

#replace your original paragraph in this text string
text = """
A ship is a large watercraft that travels the world's oceans and other sufficiently deep waterways, carrying cargo or passengers, or in support of specialized missions, such as defense, research, and fishing. Ships are generally distinguished from boats, based on size, shape, load capacity, and purpose. Ships have supported exploration, trade, warfare, migration, colonization, and science. After the 15th century, new crops that had come from and to the Americas via the European seafarers significantly contributed to world population growth.[1] Ship transport is responsible for the largest portion of world commerce.

The word ship has meant, depending on the era and the context, either just a large vessel or specifically a ship-rigged sailing ship with three or more masts, each of which is square-rigged.

As of 2016, there were more than 49,000 merchant ships, totaling almost 1.8 billion dead weight tons. Of these 28% were oil tankers, 43% were bulk carriers, and 13% were container ships.[2]
"""

#number of word to get summary size as half
countOfWords = len(text.split())

summary_text = summarizer(text, max_length=countOfWords, min_length=5, do_sample=False)[0]['summary_text']
print("\n\nYour BERT model similarity based summary is : \n",summary_text)
