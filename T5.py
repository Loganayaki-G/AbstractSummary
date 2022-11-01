#pip install transformers
#pip install pytorch
#if pytorch error then install torch
from transformers import pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization")#To use the BART model, which is trained on the CNN/Daily Mail News Dataset, 
#or
#summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")#If you want to use the t5 model (e.g. t5-base), which is trained on the c4 Common Crawl web corpus, then

#replace your original paragraph in this text string
text = """
Recently, a growing interest to Machine Learning forced the companies to focus and invest on it. Besides, this major is considered an exciting research topic for scholars in Computer Science. Machine Learning (ML) is the major of computational science, which focuses on analyzing and interpreting patterns and structures in data to enable learning, reasoning, and decision making outside of human being interaction. This essay aims to discuss and introduce concepts, methods, and steps of Machine Learning.
"""

#number of word to get summary size as half
countOfWords = len(text.split())

summary_text = summarizer(text, max_length=countOfWords, min_length=5, do_sample=False)[0]['summary_text']
print("\n\nYour T5 model similarity based summary is : \n",summary_text)
