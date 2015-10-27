# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:40:38 2015

@author: lsharma
"""

#import nltk
#nltk.download()

from nltk.corpus import stopwords

sw = stopwords.words("english")
print "Number of stop words:", len(sw)