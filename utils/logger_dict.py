from collections import OrderedDict
import csv
from genericpath import exists
import os

from utils.print_progress import progress_bar

keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1', 'ParameterScale']

class Logger_dict():
    def __init__(self, logger, save_path):
        self.dict = OrderedDict()
        self.logger = logger
        self.savepath = save_path
        if os.path.exists(os.path.join(self.savepath, 'log.csv')):
            self.init_csv()
        self.write_csv(keys)
                
    def update(self, key, value):
        self.dict[key] = value
        
    def init_csv(self):
        fileVariable = open(os.path.join(self.savepath, 'log.csv'), 'r+')
        fileVariable.truncate(0)
        fileVariable.close()
        
    def write_csv(self, x):
        with open(os.path.join(self.savepath, 'log.csv'), "a") as outfile:
            csvwriter = outfile
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(x)
        
    def print(self):
        i = 0
        values = []
        for key, value in self.dict.items():
            print(f'{key}' +'\t'+ f'{value}')
            i += 1
            values.append(value)
        self.write_csv(values)
        print()
            
    