import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import re

class DataSet(object):
    def __init__(self, headers, data):
        self.headers=headers
        self.data=data

#import data from a csv file
    def import_from_csv(self, path_to_csv):
        with open(path_to_csv, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            headers=data.pop(0)
            data=np.array(data)
        self.headers=headers
        self.data=data
        return headers, data

#Extract only certai fields (the ones in fields array)
    def extract_fields_data(self, fields):
        new_headers=[]
        indices=[]
        self.data=np.array(self.data)
        for field in fields:
            if(self.headers.index(field)==-1):
                raise RuntimeError('%s is not a valid field of data set' %field)
            else:
                new_headers.append(field)
                indices.append(self.headers.index(field))
        self.headers=new_headers
        self.data=self.data[:, indices]
        return self.headers, self.data

#plots the number most common field patterns
    def plot_most_common_patterns(self, field, number):
        if(self.headers.index(field)==-1):
            raise RuntimeError('%s is not a valid field of data set' %field)
        else:
            counter=Counter(self.data[:][self.headers.index(field)])
        labels=[]
        counter=counter.most_common(number)
        y=[]
        for element in counter:
            labels.append(element[0])
            y.append(element[1])
        fig = plt.figure()
        fig.suptitle('%d most common patterns in %s field' %(number, field))
        plt.bar(labels, y)
        plt.show()
        return


    def plot_distribution(self):
        #to be done
        return

#This function deletes all the rows where the value corresponding to field is equal to pattern
    def clear_data(self, field, pattern):
        if(self.headers.index(field)==-1):
            raise RuntimeError('%s is not a valid field of data set' %field)
        else:
            self.data=self.data[np.logical_not(self.data[:, self.headers.index(field)]==pattern)]
        return self.data

#Clears the data and keeps only the row where the pattern corresponding to field is in list_of_interest
    def clear_data_in_list(self, field, list_of_interest):
        new_data=[]
        if(self.headers.index(field)==-1):
            raise RuntimeError('%s is not a valid field of data set' %field)
        else:
            for row in self.data:
                if(row[self.headers.index(field)] in list_of_interest):
                    new_data.append(row)
        self.data=np.array(new_data)
        return self.data
