# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:51:50 2017

@author: azunre
"""

import numpy as np

# standardize all column row number for fixed input-length CNN,
# used for standardization of encoded (3-dimensional) NN data
def DataLengthStandardizerEncoded(X, max_cells):
    nsamples, nx, ny = X.shape
    if nx >= max_cells:
        X = np.delete(X, np.arange(max_cells, nx), axis=1)  # truncate
    else:
        # replicate some randomly-selected cells
        X = np.concatenate(
            (X, X[:, np.random.choice(nx, max_cells - nx), :]), axis=1)
    return X
    
# standardize a single column row number (column as list input, array output, 
# used for column-wise data-length standardization of raw data)
def DataLengthColumnStandardizerRaw(column, max_cells):
    
    RAND = False # specify whether to pick a random subset of rows, or the first rows 
    
    if np.any(column[column.notnull()]):
        column = column[column.notnull()] # remove all nulls
    else:
        column = column.fillna('NaN')
    nrows = len(column)
    
    if nrows >= max_cells:
        column = np.asarray(column)
        if(RAND):
            column = column[np.random.choice(nrows,max_cells)] # random may not be preferred in this case
        else:
            column = np.delete(column,np.arange(max_cells,nrows))
    else:
        # replicate some randomly-selected cells
        column = np.asarray(column)
        column = np.concatenate([column,column[np.random.choice(nrows, max_cells - nrows)]])
        
    return column[np.newaxis].T

# standardize all column row number (used for whole raw dataset, input should
# a pandas dataframe, output also returns a flag that alerts system of inaccuracies
# for 'statistical' (ordinal, categorical, etc) variable prediction due to truncation)
def DataLengthStandardizerRaw(data, max_cells):
    nx, ny = data.shape
    out = DataLengthColumnStandardizerRaw(data.iloc[:,0],max_cells)
    for i in np.arange(1,ny):
        out = np.concatenate((out,DataLengthColumnStandardizerRaw(data.iloc[:,i],max_cells)),axis=1)
    return out
