import pandas as pd
import numpy as np
import collections
import os
import nems.xforms as xforms
import nems.modelspec as ms

'''
collection of functions to extract and parse data from a batch of fitted cells
'''

def flatten_dictionary(d, parent_key='', sep='_'):
    # TODO check that this works
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flat_dict_to_df(d):
    # TODO this is a placeholder. better implement condiering flattened dictionary, should contain pointer to modelspec files
    df = pd.DataFrame.from_dict({(i, j): d[i][j]
                            for i in d.keys()
                            for j in d[i].keys()},
                           orient='index')
    return df

def load_single_modspec(cellid, batch, modelname):
    '''
    load the modelspec of an oddball recordign including SSA related metrics within the metadata
    :param cellid: str the name of the cell
    :param batch: num, for oddbal data it is 296
    :param modelname: str a string of characters describing the model
    :return:
    '''
    source_dir = '/auto/users/mateo/oddball_results/{0}/{1}/{2}/'.format(
        batch, cellid, modelname)
    xforms, ctx = xforms.load_analysis(source_dir, eval_mode=False)
    modelspecs = ctx['modelspecs']

    return modelspecs

def read_SI(modelspecs, as_DF=False):
    '''
    reads a modelspec and returns the ssa index values as a dictionary or a dataframe
    :param modelspecs: a modelspec data structure
    :param as_DF: bool, if true returns a dataframe
    :return: a data structure with valued of ssa index organized by actual/predicted, Jitter On/Off ...ToDo any other category
             returns values as nested dictionary, or whatever the default of modelspecs is. or a long format dataframe
    '''

    meta = modelspecs[0][0]['meta']
    SI = meta['SSA_Index ']
    if as_DF == True:
        SI = flatten_to_df(SI)

    return SI


def flatten_to_df(nested_dict):
    # ToDo implement
    raise NotImplementedError('make mateo implement this this shit lazy ass')