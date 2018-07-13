import joblib as jl
import pandas as pd
import numpy as np
import seaborn as sns
import oddball_DF as odf
import matplotlib.pyplot as plt

df = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal_jackknife')

modelname1 = 'odd_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
modelname2 = 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
pair1 = [modelname1, modelname2]
title1 = 'stp model with stim onset'

modelname3 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
modelname4 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
pair2 = [modelname3, modelname4]
title2 = 'stp model with stim envelope'

modelpairs = [pair1, pair2]
subplot_titles = [title1, title2]

# color maps to response and prediction
colors = ['green', 'red']

parameter = 'SSA_index'
stream = 'cell'

# populatio quality filter
metric = 'r_test'
metric = 'activity'
threshold = 2

# filters for plotting SSA index
Jitter = 'Jitter_Both'
resp_pred = 'pred'

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
axes = np.ravel(axes)
# make the subplot represent the model architecture
for ii, (ax, modelpair) in enumerate(zip(axes, modelpairs)):

    # maeke resp pred be the color
    for color, repr in zip(colors, ['resp', 'pred']):

        DF = df.copy()

        # recording quality control, selects by activity level
        quality_filtered = odf.filter_df_by_metric(DF, metric=metric, threshold=threshold)

        # defiens filters
        ff_modelname = quality_filtered.modelname.isin(modelpair)
        ff_parameter = quality_filtered.parameter == parameter
        ff_stream = quality_filtered.stream == stream
        ff_Jitter = quality_filtered.Jitter == Jitter
        ff_resp_pred = quality_filtered.resp_pred == repr

        filtered = quality_filtered.loc[ff_modelname & ff_parameter & ff_stream & ff_Jitter & ff_resp_pred, :]
        # check and eliminate duplicates
        print('duplicates: {}'.format(filtered.duplicated(['cellid', 'modelname']).any()))
        filtered = filtered.drop_duplicates(subset=['cellid', 'modelname'])

        # gets the mean of the jackknifes

        meaned = odf.collapse_jackknife(filtered)

        pivoted = meaned.pivot(index='cellid', columns='modelname', values='value')
        cols = pivoted.columns
        pivoted.plot(cols[0], cols[1], kind='scatter', color=color, ax=ax, label='{}'.format(repr), picker=True)

    ax.set_title(subplot_titles[ii])
    ax.set_xlabel('SI from full sig')
    ax.set_ylabel('SI from jackknife mean')
fig.suptitle('180711_SI_full_vs_jackknife.py')
