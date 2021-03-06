import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sts
import seaborn as sns
import matplotlib.pyplot as plt

#### ploting parameters

# resp_pred = 'pred'
# modelname = 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

modelname1 = 'odd1_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
modelname2 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
modelname3 = 'odd1_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
modelname4 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'

modelnames = [modelname1, modelname2, modelname3, modelname4] #, modelname3, modelname4]

parameter = 'SSA_index' # right now only works with SSA_index

stream = ['f1', 'f2', 'cell']
stream = ['cell']

Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0.15

# activity level filter
# metric = 'activity'
# threshold = 0

# compare no-jackknife with jackknife

compare_jk = True # if true Jitter must be a list of a single value



######## script starts here
'''
dimentions of the plot 
subplots row: Jitter status 
subplots col: stream  
color: model

unfortunately now i know that x and y must be independent columns, bummer
x ax = recorded          
y ax = predicted
'''

loaded = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal_jackknife')

def stp_plot(parameter=parameter, modelnames=modelnames, Jitter=Jitter, stream=stream, threshold=threshold):
    DF = loaded.copy()
    DF = odf.collapse_jackknife(DF)

    # filter by goodnes of fit
    quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold= threshold)

    # filter by parameters

    ff_param = quality_filtered.parameter == parameter
    ff_model = quality_filtered.modelname.isin(modelnames)
    ff_jitter = quality_filtered.Jitter.isin(Jitter)
    ff_stream = quality_filtered.stream.isin(stream)
    #ff_resppred = quality_filtered.resp_pred == resp_pred

    filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream, :]

    more_parms =  ['modelname', 'Jitter', 'stream', 'cellid']
    pivot_by = 'resp_pred'
    values = 'value'

    tidy = odf.make_tidy(filtered,pivot_by, more_parms, values)
    # split the modelname into two subsets, model architecture (null or alternative model) and full fit or jackknife fit
    tidy['model_architecture'] = [arch.split('_')[1] for arch in tidy.modelname] # get model structure
    tidy['jackknife'] = [jkn.split('_')[-3] for jkn in tidy.modelname] # get jacknife or not

    # renames the new columns values for better labeling in plots

    tidy = tidy.replace({'fir2X15': 'w/o_STP', 'stp2': 'w_STP', 'basic-nftrial': 'full_calc', 'si-jk': 'jackknifes'})


    if compare_jk is False:
        g = sns.FacetGrid(tidy, row='Jitter', col='stream', hue='modelname')
        g = (g.map(plt.scatter, 'resp', 'pred', edgecolor="w", alpha=0.8).add_legend())
    elif compare_jk is True:
        g = sns.FacetGrid(tidy, row='model_architecture', col='stream', hue='jackknife')
        g = (g.map(plt.scatter, "resp", "pred", edgecolor="w", alpha=0.8).add_legend())

    fig = g.fig
    fig.suptitle('{} value prediction'.format(parameter))
    return DF

DF = stp_plot()
