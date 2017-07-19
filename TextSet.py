import csv
import sys
import pandas as pd
from collections import Counter
import numpy as np
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence


#import data from a csv file
def import_from_csv(path_to_folder):
    train_data=pd.read_csv(path_to_folder+'/training_variants')
    train_text=pd.read_csv(path_to_folder+'/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    test_data=pd.read_csv(path_to_folder+'/test_variants')
    test_text=pd.read_csv(path_to_folder+'/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    train = pd.merge(train_data, train_text, how='left', on='ID').fillna('')
    test = pd.merge(test_data, test_text, how='left', on='ID').fillna('')
    return train, test

def format_text(data_frame):
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
