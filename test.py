#!/usr/bin/python

import pandas as pd
import numpy as np

def print_matrix(mat1, filename='output'):
    df = pd.DataFrame(mat1)
    writer = pd.ExcelWriter('/Users/Gabriele/Desktop/Poli/OMGP_python/outputs/%s.xlsx' % (filename), engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')


def read_matrix(filename):
    df = pd.read_excel('%s.xlsx' % (filename))
    return np.matrix((df))




