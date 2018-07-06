import pandas as pd
import numpy as np
import collections
import os
import nems.xforms as xforms
import nems.modelspec as ms
import nems_db.db as nd
import warnings
import itertools as itt


'''
collection of functions to extract and parse data from a batch of fitted cells
'''

### base low level functions

def get_source_dir(cellid, batch, modelname):

    source_dir = '/auto/users/mateo/oddball_results/{0}/{1}/{2}/'.format(
        batch, cellid, modelname)

    return source_dir


def load_single_modspec(cellid, batch, modelname):
    '''
    load the modelspec of an oddball recordign including SSA related metrics within the metadata
    :param cellid: str the name of the cell
    :param batch: num, for oddbal data it is 296
    :param modelname: str a string of characters describing the model
    :return:
    '''
    source_dir = get_source_dir(cellid, batch, modelname)

    _, ctx = xforms.load_analysis(source_dir, eval_model=False)
    modelspecs = ctx['modelspecs']

    return modelspecs

def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "&" + subkey, subvalue
            elif isinstance(value, (int, float, str, )):
                yield key, value
            else:
                mesg = 'object at {} is type {}'.format(key, type(value))
                warnings.warn(Warning(mesg))
                yield key, value

    return dict(items())


def dict_to_df(nested_dict, column_names=None):
    '''
    transforms a structure of nested dictionaries into a dataframe in long format.
    Columns correspond to nesting levels, and values corespond to dictionary keys ( for the parameter columns ) or
    values from the actual data in the lowest level of the dictionary

    :param nested_dict: {key: {key: val}, ...} a data structure of nested dictionary

    :param column_names: [str,...] a list of strings with names for the df columns (equivalent to dict nesting level)

    :return: a dataframe
    '''
    # recursively navigates the nested dictionaries and concatenates keys as a string with a "&" separator
    flat_dict = flatten_dict(nested_dict)

    # checks the shape of de dictionary ToDo iplement

    # creates a df from the flattened dicionary, using dict keys as row indexes
    df = pd.DataFrame([flat_dict]).transpose()

    # creates a multiindex spliting the compound keys of the flattened dict
    df.index = pd.MultiIndex.from_tuples(df.index.str.split('&').tolist())

    # renames the multyindex level and values columm
    if isinstance(column_names, list):
        df.index.rename(column_names,level=None, inplace=True)
    elif column_names is None:
        mesg = 'give names to the columns to avoid future headaches!'
        warnings.warn(Warning(mesg))
    df.columns = ['value']

    # turns resets the multinindex into columns
    df = df.reset_index()

    return df


def get_from_meta(modelspecs, key, as_DF=False, column_names=None):

    '''
       reads a modelspec and returns the ssa index values as a dictionary or a dataframe
       :param modelspecs: a modelspec data structure
       :param as_DF: bool, if true returns a dataframe
       :return: a data structure with valued of ssa index organized by actual/predicted, Jitter On/Off ...ToDo any other category
                returns values as nested dictionary, or whatever the default of modelspecs is. or a long format dataframe
       '''

    meta = modelspecs[0][0]['meta']
    metrics = meta[key]
    if as_DF == True:
        metrics = dict_to_df(metrics, column_names)

    return metrics

def get_stp_values(modelspecs):
    '''
    finds the Tau and U values within the modelspecs data structure

    :param modeslpecs: and oddball experiment modeslpecs

    :return:  a dict of dicts of lists
             {'Tau': {'f1':[float,...], 'f2':[float,...]} ,
             'U': {'f1':[float,...], 'f2':[float,...]}}
        dict 1: tau or U
        dict 2: frequency 1 or two
        list: repetition of the same value due to jackknifing
    '''

    # init first dict layer
    stp = dict.fromkeys(['tau', 'u'])
    for outerkey, _ in stp.items():
        # init second dict layer
        stp[outerkey] = dict.fromkeys(['f1', 'f2'])
        # init lists
        for innerkey, _ in stp[outerkey].items():
            stp[outerkey][innerkey] = list()

    # iterate over each of the estimation subsets
    for jackknife in modelspecs:
        # takes the values correspondign to the STP module
        phi = jackknife[0]['phi']
        for key, val in phi.items():
            stp[key]['f1'].append(val[0])
            stp[key]['f2'].append(val[1])

    return stp


### script like functions for single oddball experiments

def single_specs_to_DF(cellid, batch, modelname):
    '''
    MLE. 05, July, 2018

    pull all relevant values form modelspecs, organizes in a long format dataframe

    :param cellid: str single cell identifier

    :param batch: int, 296 for oddball experiments

    :param modelname: str, chain of str corresponding to model architecture.

    :return: a data frame in long format with  all the pertinent data.

    '''
    # loads all metadata
    modelspecs = load_single_modspec(cellid, batch, modelname)

    frames = list()

    # this list defines my specific the keys of the metadata to be pulled
    parameters = ['SSA_index', 'activity', ]
    for parameter in parameters:
        # generates a df for each parameter
        parm_DF = get_from_meta(modelspecs, key=parameter, as_DF=True, column_names=['Jitter', 'act_pred', 'stream'])
        # adds parameter column
        parm_DF['parameter'] = parameter

        frames.append(parm_DF)

    # gets the values of fitted stp parameters

    stp_DF = dict_to_df(get_stp_values(modelspecs), column_names=['parameter', 'stream'])
    stp_DF['Jitter'] = 'Jitter_Both'
    frames.append(stp_DF)

    # add more frames to the DF ??


    # concatenates frames

    DF = pd.concat(frames, sort=True)

    return DF


### script like fuinctions for single oddball experiments

def batch_specs_to_DF(batch, modelnames):
    '''
    organizes relevant metadata from all cells in a batch into a dataframe

    :param batch: int, batch number, 296 in case of oddball
    :param modelname: list of modelnames, each modelname is a chain of string specifying the model architecture.
    :return: a data frame in long format. all numerical values are in a single column, all other columns correspond to tags

    '''

    # get a list of the cells in the batch
    batch_cells = nd.get_batch_cells(batch=batch).cellid
    # gets the single cells data frames and adds cellid
    single_cell_dfs = list()
    for cellid, modelname in itt.product(batch_cells, modelnames):
        try:
            df = single_specs_to_DF(cellid, batch, modelname)
        except FileNotFoundError:
            mesg = 'file not found: {}/xfspec.json'.format(get_source_dir(cellid, batch, modelname))
            warnings.warn(Warning(mesg))
            continue
        except:
            mesg = 'WTF just happened with {}'.format(cellid)
            warnings.warn(Warning(mesg))
            continue


        df['cellid'] = cellid
        single_cell_dfs.append(df)

    DF = pd.concat(single_cell_dfs, sort=True)

    # adds modelname
    DF['modelname'] = modelname

    return DF



### data frame manipulations

def collapse_jackknife(DF, func=np.mean):
    '''
    collapses jackknife repeated values using the defined function

    :param DF: a pandas DF in long format, with numerical values (int, float ...)
               or groups of numerical values (list, nparr...)
    :param func: a function able to work on groups of numerical values, e.g. np.mean
    :return: DF with collapsed groups of values
    '''

    out_df = DF.copy()
    out_df.value.apply(func, inplace=True)

    return out_df
