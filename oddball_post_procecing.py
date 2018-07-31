import pandas as pd
import numpy as np
import nems.xforms as xforms
import nems_db.db as nd
import warnings
import copy

'''
collection of functions to extract and parse data from a batch of fitted cells
'''


#### base low level functions ####

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
            else:
                yield key, value

    return dict(items())


def unflatten_dict(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split("&")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


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
        df.index.rename(column_names, level=None, inplace=True)
    elif column_names is None:
        mesg = 'give names to the columns to avoid future headaches!'
        warnings.warn(Warning(mesg))
    df.columns = ['value']

    # turns resets the multinindex into columns
    df = df.reset_index()

    return df


def df_to_dict(DF):
    if len(DF.columns) == 1:
        if DF.values.size == 1: return DF.values[0][0]
        return DF.values.squeeze()
    grouped = DF.groupby(DF.columns[0])
    d = {k: df_to_dict(g.ix[:, 1:]) for k, g in grouped}
    return d


def df_vals_tolist(DFs):
    '''
    froma a list of shape consisten df, retunr a df whos values are a list of the values of each independent df
    the returned df has the same shape and structure a the original DF.
    :param DFs:
    :return:
    '''
    df_list = list()
    for df in DFs:
        indexed = df.set_index(keys=[col for col in df.columns if col != 'value'])
        df_list.append(indexed)

    merged = pd.concat(df_list, axis=1, join='outer', sort=True)
    matrix = merged.values.tolist()
    merged['as_list'] = matrix

    list_DF = merged['as_list']
    newDF = list_DF.reset_index()

    newDF = newDF.rename(columns={'as_list': 'value'})

    return newDF


def swap_struct_levels(list_of_dicts):
    # todod for each nested dict, flat dictionary
    flat_dicts = [flatten_dict(d) for d in list_of_dicts]
    out_dict = copy.copy(flat_dicts[0])

    for fkey in out_dict.keys():
        # for each unit compounded key, pool all values in list
        val_list = [fdict[fkey] for fdict in flat_dicts]
        out_dict[fkey] = val_list

    out_dict = unflatten_dict(out_dict)

    return out_dict


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

    meta = modelspecs[0][0]['meta']

    # backwards_compatibility. old vs new kewyword paradigm
    kws_old = meta['modelspecname'].split('_')
    kws_new = meta['modelspecname'].split('-')
    if len(kws_old) != 1 and len(kws_new) ==1:
        # old keywords
        kws = kws_old
        stp_kw = 'stp2'

    elif len(kws_old) == 1 and len(kws_new) !=1:
        # new keywords
        kws = kws_new
        stp_kw = 'stp.2'
    else:
        raise ValueError("cannot split modelspecs into keyword with '-' or '.' ")

    # init first dict layer
    stp = dict.fromkeys(['tau', 'u'])
    for outerkey, _ in stp.items():
        # init second dict layer
        stp[outerkey] = dict.fromkeys(['f1', 'f2'])
        # init lists
        for innerkey, _ in stp[outerkey].items():
            stp[outerkey][innerkey] = list()

    # looks for the stp in the model architecture keywords
    if stp_kw in kws:
        position = kws.index(stp_kw)
    else:
        # returns an empty dictionary
        for outerkey, innerdict in stp.items():
            for innerkey in innerdict.keys():
                stp[outerkey][innerkey] = np.nan

        return stp

    # iterate over each of the estimation subsets
    for jackknife in modelspecs:
        # takes the values correspondign to the STP module
        phi = jackknife[position]['phi']
        for key, val in phi.items():
            stp[key]['f1'].append(val[0])
            stp[key]['f2'].append(val[1])

    return stp


def get_est_val_sets(modelspecs):
    meta = modelspecs[0][0]['meta']

    if 'est_set' not in meta.keys() and 'val_set' not in meta.keys():
        mesg = 'est val subsets not defined, asuming full recording for both'
        warnings.warn(Warning(mesg))

        meta['est_set'] = 'jal'
        meta['val_set'] = 'jal'

    elif 'est_set' in meta.keys() and 'val_set' not in meta.keys():
        raise ValueError('est set defined, val set undefined')
    elif 'est_set' not in meta.keys() and 'val_set' in meta.keys():
        raise ValueError('val set defines, est set undefined ')
    elif 'est_set' in meta.keys() and 'val_set' in meta.keys():
        pass
    else:
        raise ValueError('WTF? the universe is broken')

    est_val_sets = {'est_set': meta['est_set'], 'val_set': meta['val_set']}

    return est_val_sets


def get_corrcoef(modelspecs):
    meta_key_list = ['r_test', 'se_test', 'r_floor', 'mse_test', 'll_test',
                     'r_fit', 'se_fit', 'r_ceiling', 'mse_fit', 'll_fit',
                     'jk_r_test']

    meta = modelspecs[0][0]['meta']

    corrcoef_dict = dict.fromkeys(meta_key_list)

    for key in corrcoef_dict.keys():
        corrcoef_dict[key] = meta[key]

    return corrcoef_dict


#### higher level API for lazy pulls ####

def get_si(ctx):
    modelspecs = ctx['modelspecs']

    SI = get_from_meta(modelspecs, 'SSA_index', as_DF=True, column_names=['Jitter', 'resp_pred', 'stream'])

    return SI


def get_ra(ctx):
    modelspecs = ctx['modelspecs']

    RA = get_from_meta(modelspecs, 'activity', as_DF=True, column_names=['Jitter', 'resp_pred', 'stream'])

    return RA


#### script like functions for single oddball experiments ####

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
        parm_DF = get_from_meta(modelspecs, key=parameter, as_DF=True, column_names=['Jitter', 'resp_pred', 'stream'])
        # adds parameter column
        parm_DF['parameter'] = parameter

        frames.append(parm_DF)

    # gets the values of fitted stp parameters

    stp_DF = dict_to_df(get_stp_values(modelspecs), column_names=['parameter', 'stream'])
    # stp_DF['Jitter'] = 'Jitter_Both'
    frames.append(stp_DF)

    # get correlation coefficient values

    corrcoef_DF = dict_to_df(get_corrcoef(modelspecs), column_names=['parameter'])
    frames.append(corrcoef_DF)

    # add more frames to the DF ??

    # concatenates frames

    DF = pd.concat(frames, sort=True)

    # add tags to the whole DF
    est_val_sets = get_est_val_sets(modelspecs)

    for key, val in est_val_sets.items():
        DF[key] = val

    return DF


#### script like fuinctions for batches ####

def batch_specs_to_DF(batch, modelnames):
    '''
    organizes relevant metadata from all cells in a batch into a dataframe

    :param batch: int, batch number, 296 in case of oddball
    :param modelname: list of modelpairs, each modelname is a chain of string specifying the model architecture.
    :return: a data frame in long format. all numerical values are in a single column, all other columns correspond to tags

    '''

    # get a list of the cells in the batch
    batch_cells = nd.get_batch_cells(batch=batch).cellid
    # gets the single cells data frames and adds cellid

    models_DF = list()
    no_file_error = list()
    unexpected_error = list()

    for modelname in modelnames:

        cells_in_model = list()

        for cellid in batch_cells:
            try:
                df = single_specs_to_DF(cellid, batch, modelname)
            except FileNotFoundError:
                # mesg = 'file not found: {}/xfspec.json'.format(get_source_dir(cellid, batch, modelname))
                # warnings.warn(Warning(mesg))
                no_file_error.append('{} {}'.format(modelname, cellid))
                continue
            except:
                mesg = 'WTF just happened with {} {}'.format(cellid, modelname)
                warnings.warn(Warning(mesg))
                unexpected_error.append('{} {}'.format(modelname, cellid))

                continue

            df['cellid'] = cellid
            cells_in_model.append(df)

            DF = pd.concat(cells_in_model, sort=True)

        # adds modelname
        # change old modelname to new one, just aesthetic
        if modelname == 'stp2_fir2x15_lvl1_basic-nftrial':
            modelname = 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

        DF['modelname'] = modelname

        DF.dropna(axis=0, subset=['value'], inplace=True)

        models_DF.append(DF)

    DF = pd.concat(models_DF, sort=True)

    DF = DF.reset_index(drop=True)

    return DF, no_file_error, unexpected_error
