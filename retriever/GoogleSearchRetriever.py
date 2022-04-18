import sys
import pickle
from urllib.parse import unquote
from googlesearch import search
from TfidfRetriever import TfidfRetriever, HierarchicalTfidf
import numpy as np
import requests
import json
import urllib
import time
import wiki_stemmer

class ScrapGoogleSearchRetriever:
    """
    This uses the python library google, this library does a web scrape and thus only works for small requests
    """
    def __init__(self, docs, k=5):
        """
        :param docs: dictionary form of Wikipedia
        :param k:  number of articles to return
        """
        self.k = k
        self.docs = docs

    def get_topk_docs(self, query):
        urls = search('site:ar.wikipedia.org' +" " +str(query), stop=self.k, domains=["ar.wikipedia.org"])
        article_titles = []
        for url in urls:
            article_titles.append(unquote(url[30:]).replace("_"," "))
        top_docs = []
        for title in article_titles:
            if title in self.docs:
                top_docs.append(self.docs[title])
            else:
                top_docs.append([""])
        return top_docs[:min(self.k,len(top_docs))]


class ApiGoogleSearchRetriever:
    """
    We call the official Google Custom Search API, need to obtain API Key first and CSE, $5 per 5000requests
    """
    def __init__(self, docs, k, base_retriever):
        """
        :param docs: dictionary form of Wikipedia
        :param k:  number of articles to return
        """
        self.k = min(k, 10) # API has max 10 results
        self.docs = docs
        id = 0
        self.base_retriever = base_retriever
        # custom search engine ID, need to create on cloud shell
        self.CSE = "58eb65c4e6c2f7475"
        # API KEY for custom search
        self.API_KEY = "AIzaSyCiaYwBU6fziCgkSkRFFO0hBKIfUevctrY"

    def get_topk_docs_scores_merged(self, query):
        original_query = query
        query = urllib.parse.quote_plus(query)
        DATA = {}
        id = 0
        while "items" not in DATA:
            if id > 0:
                print("retrieving failed ... \n Retrying")
                time.sleep(0.5)
            if id == 30:
                break
            url = "https://www.googleapis.com/customsearch/v1/siterestrict?q=" + str(query) + "&cx=" + self.CSE \
                  + "&num=" + str(self.k) + "&siteSearch=ar.wikipedia.org&key=" + self.API_KEY
            S = requests.Session()
            R = S.get(url=url, verify=False)
            DATA = R.json()
            id += 1
        article_titles = []
        if "items" not in DATA:
            return None, None
        for title in DATA["items"]:
            title_fixed = title['title'].replace(" - ويكيبيديا", "")
            if title_fixed in self.docs:
                article_titles.append(title_fixed)
        top_docs = []
        paragraphs = {}

        for title in article_titles[:min(self.k, len(article_titles))]:
            paragraphs[title] = ""

        for title in article_titles[:min(self.k, len(article_titles))]:
            if title in self.docs:
                for par in self.docs[title][:15]:
                    paragraphs[title] += par

        for title in article_titles[:min(self.k, len(article_titles))]:
            if len(paragraphs[title]) < 20:
                continue
            top_docs.append(paragraphs[title])

        r2 = TfidfRetriever(top_docs, len(top_docs), 4)
        top_docs, docs_scores = r2.get_topk_docs_scores(original_query)
        assert(len(top_docs) == len(docs_scores))
        return top_docs, docs_scores

    def get_topk_docs_scores(self, query):
        original_query = query  # Should be cleaned and stemmed
        query = urllib.parse.quote_plus(query)
        DATA = {}
        id = 0
        while "items" not in DATA:
            if id > 0:
                print("retrieving failed ... \n Retrying")
                time.sleep(0.5)
            if id == 30:
                break
            url = "https://www.googleapis.com/customsearch/v1/siterestrict?q=" + str(query) + "&cx=" + self.CSE \
                  + "&num=" + str(self.k) + "&siteSearch=ar.wikipedia.org&key=" + self.API_KEY
            S = requests.Session()
            R = S.get(url=url, verify=False)
            DATA = R.json()
            id += 1
        article_titles = []
        if "items" not in DATA:
            return None, None
        for title in DATA["items"]:
            title_fixed = title['title'].replace(" - ويكيبيديا", "")
            if title_fixed in self.docs:
                article_titles.append(title_fixed)
        top_docs = []
        for title in article_titles[:min(self.k, len(article_titles))]:
            if title in self.docs:
                for par in self.docs[title][:15]:
                    if len(par) >= 50:
                        top_docs.append(par)
        r2 = TfidfRetriever(top_docs, len(top_docs), 4)
        top_docs, docs_scores = r2.get_topk_docs_scores(original_query)
        assert(len(top_docs) == len(docs_scores))
        return top_docs, docs_scores


    def get_topk_docs(self, query):
        query = urllib.parse.quote_plus(query)
        url = "https://www.googleapis.com/customsearch/v1/siterestrict?q=" + str(query) + "&cx=" + self.CSE\
              + "&num=" + str(self.k) + "&siteSearch=ar.wikipedia.org&key=" + self.API_KEY
        S = requests.Session()
        R = S.get(url=url)#, verify=False)
        DATA = R.json()
        article_titles = []
        if "items" not in DATA:
            return []
        for title in DATA["items"]:
            title_fixed = title['title'].replace(" - ويكيبيديا","")
            article_titles.append(title_fixed)

        top_docs = []
        for title in article_titles:
            if title in self.docs:
                top_docs.append(self.docs[title])
            else:
                top_docs.append([""])
        return top_docs[:min(self.k,len(top_docs))]

def test_GoogleSearchRetriever():
    wiki_data = pickle.load(open("../arwiki/arwiki.p","rb"))
    r = ApiGoogleSearchRetriever(wiki_data,5)
    print(r.get_topk_docs_scores("في اي عام كان رفع معدل النمو ل 2.2%"))
