import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#### data frame manipulations ####
def collapse_jackknife(DF, func=np.mean):
    '''
    collapses jackknife repeated values using the defined function

    :param DF: a pandas DF in long format, with numerical values (int, float ...)
               or groups of numerical values (list, nparr...)
    :param func: a function able to work on groups of numerical values, e.g. np.mean
    :return: DF with collapsed groups of values
    '''

    out_df = DF.copy()
    out_df['value'] = out_df.value.apply(func)

    return out_df


def update_old_format(DF):

    column_map = {'Jitter': 'Jitter',
                  'model_name': 'modelname',
                  'values': 'value'}

    DF = DF.rename(columns=column_map)

    value_map = {'On': 'Jitter_On',
                 'Off': 'Jitter_Off',
                 'stream0': 'f1',
                 'stream1': 'f2',
                 'actual': 'resp',
                 'predicted': 'pred',
                 'env100e_fir20_fit01_ssa': 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal',
                 'env100e_stp1pc_fir20_fit01_ssa': 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal',
                 'SI': 'SSA_index',
                 'r_est': 'r_est', # not sure what is the equivalent value with the new mse calculation
                 'Tau': 'tau',
                 'U': 'u'}

    DF = DF.replace(to_replace = value_map)

    return DF


def relevant_from_old_DF(df):

    DF = update_old_format(df)

    new_modelnames = ['odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal',
                      'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal']

    DF = DF.loc[DF.modelname.isin(new_modelnames), : ]

    value_map = {'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal': 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal_old',
                 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal': 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal_old'}

    DF = DF.replace(to_replace=value_map)

    return DF



def filter_df_by_metric(DF, metric='r_test', threshold=0):

    '''
    returnts a DF with only those cellid/modelname combinations, in which a metric criterion is achieved
    it is recomended that the DF comes from a signle model.

    :param DF: and oddball batch summary DF
    :param metric: the parameter value to be considered for selection
    :threshold: the value the metric has to obe over
    :return: a DF of the same structure as the original, including only good cells
    '''

    wdf = DF.copy()
    # creates a unique file identifier based on cellid and modelname.
    wdf['unique_ID'] = ['{}@{}'.format(cell, model) for cell, model in zip(wdf.cellid, wdf.modelname)]

    # select the metric value and the unique_ID from the original DF
    ff_metric = wdf.parameter == metric
    metric_DF = wdf.loc[ff_metric, ['unique_ID', 'value']]

    # from the metric DF select the values that fullfill the criterion
    ff_criterion = metric_DF.value >= threshold
    good_files =  metric_DF.loc[ff_criterion, 'unique_ID'].unique()

    # set off cellid modelnames to be kept
    ff_goodfiles = wdf.unique_ID.isin(good_files)
    ff_badfiles = ~ff_goodfiles

    # gets the original DF containing only the goodfiles

    df =DF.loc[ff_goodfiles,:]

    return df


def goodness_of_fit(DF, metric='r_test', modelnames = None, plot=False):
    '''
    simply retunrs an orderly DF withe cellid as index, modelnames as column and a goodness of fit metric as values

    :param DF: and oddball batch summary DF
    :param metric: the parameter value to be considered in the population mean
    :param modelnames: the list of models to consider, if None considres everything
    :plot: boolean, to plot or not to plot. box plot
    :return: pivoted DF
    '''

    ff_metric = DF.parameter == metric

    if modelnames is None:
        modelnames = DF.modelname.unique()
    elif isinstance(modelnames,list):
        pass

    ff_model = DF.modelname.isin(modelnames)


    filtered = DF.loc[ff_metric & ff_model, ['cellid', 'modelname', 'value']]
    print('duplicates: {}'.format(filtered.duplicated(['cellid', 'modelname']).any()))
    filtered.drop_duplicates(subset=['cellid', 'modelname'], inplace=True)

    pivoted = filtered.pivot(index='cellid', columns='modelname', values='value')

    if plot is True:

        ax = sns.boxplot(data=pivoted)
        ax = sns.swarmplot(data=pivoted, color=".25")


    return pivoted



