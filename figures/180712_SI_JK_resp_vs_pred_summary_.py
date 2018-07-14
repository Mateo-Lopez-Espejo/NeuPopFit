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
from oddball_DF import make_tidy

modelname1 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
modelname2 = 'odd1_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'

modelnames = [modelname1, modelname2]

parameter = 'SSA_index' # right now only works with SSA_index

stream = ['f1', 'f2', 'cell']
#stream = ['cell']

Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
#Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0.15

# activity level filter
# metric = 'activity'
# threshold = 0



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
    quality_filtered = odf.filter_df_by_metric(DF, metric=metric, threshold= threshold)

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

    g = sns.FacetGrid(tidy, row='Jitter', col='stream', hue='modelname')
    g.map(plt.scatter, 'resp', 'pred')
    #sns.factorplot(x='resp', y='pred', hue='modelname', data=tidy, row='Jitter', col='stream', kind='point')
    return DF

DF = stp_plot()
