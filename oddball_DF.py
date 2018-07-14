import copy

import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_post_procecing as opp

#### data frame manipulations ####
def collapse_jackknife(DF, output='mean'):
    '''
    calculates the mean and standard error from a st of jacknife values, returns what is specified by the parameter outpu

    :param DF: a pandas DF in long format, with numerical values (int, float ...)
               or groups of numerical values (list, nparr...)
    :param output: str 'mean', 'error'
    :return: DF with collapsed groups of values
    '''
    out_df = DF.copy()
    if output == 'mean':
        out_df['value'] = out_df.value.apply(np.mean)
    elif output == 'error':
        def jk_ee(jks):
            return np.mean(jks) * np.sqrt(len(jks) - 1)
        out_df['value'] = out_df.value.apply(jk_ee)
    else:
        raise ValueError ("invalid output value: {}, use 'mean' or 'error'".format(output))

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
    initial_num = len(wdf.unique_ID.unique())

    # select the metric value and the unique_ID from the original DF
    ff_metric = wdf.parameter == metric

    if metric == 'activity':
        # todo figure out why some activity values are Nan Or Inf
        ff_jitter = wdf.Jitter == 'Jitter_Both'
        ff_resp_pred = wdf.resp_pred == 'resp'
        # get the min value of activity for each frequency

        filtered = wdf.loc[ff_metric & ff_jitter & ff_resp_pred, :]

        pivoted = filtered.pivot(index='unique_ID', columns='stream', values='value')
        act_min = pivoted.min(axis=1, skipna=False)
        # excludese inf and NaN
        noinf = act_min.replace([np.inf, -np.inf], np.nan)
        nonan = noinf.dropna()
        metric_DF = nonan.reset_index()
        metric_DF.columns = ['unique_ID', 'value']
    else:
        metric_DF = wdf.loc[ff_metric, ['unique_ID', 'value']]

    # cludge: since value can be lists from jackknifes, takes the mean instead
    metric_DF = collapse_jackknife(metric_DF)

    # from the metric DF select the values that fullfill the criterion
    ff_criterion = metric_DF.value >= threshold
    good_files =  metric_DF.loc[ff_criterion, 'unique_ID'].unique()

    # how many cells are kept
    final_num = len(good_files)
    print('filtered: holding {} values from initial {}'.format(final_num, initial_num))

    # set off cellid modelpairs to be kept
    ff_goodfiles = wdf.unique_ID.isin(good_files)
    ff_badfiles = ~ff_goodfiles

    # gets the original DF containing only the goodfiles
    df =DF.loc[ff_goodfiles,:]

    return df


def goodness_of_fit(DF, metric='r_test', modelnames = None, plot=False):
    '''
    simply retunrs an orderly DF withe cellid as index, modelpairs as column and a goodness of fit metric as values

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


def simplify_DF(DF):
    '''
    takes the complete ssa dataframe and reduces its dimentions by jitter status and stream
    also calculates mean and



    :param DF:
    :return:
    '''



    raise NotImplementedError('just to it')


def make_tidy(DF, pivot_by=None, more_parms=None, values='value'):
    # todo implement make tidy by a signle column, it should be easier.
    if pivot_by is None:
        raise NotImplementedError('poke Mateo')

    if more_parms is None:
        more_parms = [col for col in DF.columns if col!=values]

    # sets relevant  indexes
    newindexes = copy.copy(more_parms)
    newindexes.append(pivot_by)
    indexed = DF.set_index(newindexes)
    # holds only the value column
    indexed = pd.DataFrame(index=indexed.index, data=indexed[values])
    # checks for duplicates
    if indexed.index.duplicated().any():
        raise ValueError("Index contains duplicated entries, cannot reshape")

    # pivots by unstacking, get parameter columns by reseting
    tidy = indexed.unstack([pivot_by]).reset_index(col_level=pivot_by)
    # cleans unnecessary multiindex columns
    tidy.columns = tidy.columns.droplevel(0)
    return tidy


def eyeball_outliers():
    outliers = ('chn022c-a1', # seems like an inhibitory neuron, maybe SI issue solves if calculated with after stim silence
                'chn019a-a1', # outright not responsive, barely showing any spikes in deviant events for one channels
                'chn019a-d1', # either late or ofset response, not enoughe deviant responses... unresponsive cell?
                'chn019a-c1', # extreamply unresponsive and unreliable
                'chn063b-d1', # noise... is the an alignment priblem?
                'chn006a-b1', # suppresed by sound
                'gus016c-c2', # this is the real WTF, check by eye
                'chn016c-c1', # noise, slighty suppresive
                'chn073b-b2', # non responsive? some spont
                'chn008b-c2', # I dont understa why it has a negative SI value, check SI calc.
                'chn016b-d1', # again not that bad, f1 dev has some nice offset activity
                'chn062f-a2') # this one is not so bad, by modal response. not sure why model wont capture SI

    outliers = np.asarray(outliers)

    return outliers