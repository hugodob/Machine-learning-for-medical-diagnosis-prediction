import csv
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import re
csv.field_size_limit(500 * 1024 * 1024)

class TextSet(object):
    def __init__(self, headers, data):
        self.headers=headers
        self.data=data

    def import_from_csv(self, path_to_csv):
        with open(path_to_csv, 'r') as f:
            reader = csv.reader((line.replace('||', '|') for line in f), delimiter='|')
            data = list(reader)
            headers=data.pop(0)
            data=np.array(data)
        self.headers=headers
        self.data=data
        return headers, data
