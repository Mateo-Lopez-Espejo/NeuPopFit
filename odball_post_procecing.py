import pandas as pd
import numpy as np
import collections

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




