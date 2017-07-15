import csv
import sys
from collections import Counter
import numpy as np
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
csv.field_size_limit(500 * 1024 * 1024)

class TextSet(object):
    def __init__(self, headers, text):
        self.headers=headers
        self.text=text
#import data from a csv file
    def import_from_csv(self, path_to_csv):
        with open(path_to_csv, 'r') as f:
            reader = csv.reader((line.replace('||', '|') for line in f), delimiter='|')
            text = list(reader)
            headers=text.pop(0)
            text=np.array(text)
        self.headers=headers
        self.text=text
        return headers, text

    def format_text(self):
        for i in range(self.text.shape[0]):
            self.text[i][1]=self.text[i][1].lower()
            self.text[i][1]= re.sub(r'[^\w\s]','',self.text[i][1])
        #Important to shuffle now if we want to have consistent results
        np.random.shuffle(self.text)
        return self.text

    def format_labeled_sentences(self):
        sentences=[]
        self.format_text()
        for i in range(self.text.shape[0]):
            sentences.append(LabeledSentence(self.text[i][1].split(),[str(i)]))
        return sentences
