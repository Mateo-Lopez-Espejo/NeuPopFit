import joblib as jl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scikits.bootstrap as bs
import oddball_plot as op

'''Given the overfitting of the model and the prediction to silence, we can use the goodness of fit for the SSA index
as as proxi for the model perfomance'''
#### ploting parameters #####

all_modelnames =np.asarray(['odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal',
                           'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal',
                           'odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal',
                           'odd1_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'])

modelnames = [2,3]
jitter = 'Jitter_Both'
stream = 'cell'

which_plot = 'scatter' #'delta'


##### scipt starts here ######
# Imports Old Df containting both the model w/o and with STP. Unfortunately, the model without STP was never done
# with the paradigm of fitting and evaluating with different subsets (randon vs regular intervals) of the trials.

DF = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal')

#creates new column with model (2) and resp_pred (2) combinations as a single variable
DF['SI_source'] = ['{}_{}'.format(model.split('_')[1], ap) for model, ap in zip(DF.modelname, DF.resp_pred)]

# Filters for SSA index values, calculated for the whole cell (frequency independent) and, for the poled data of jitter on and of

modelnames = all_modelnames[modelnames]
ff_model = DF.modelname.isin(modelnames)
ff_parameter = DF.parameter == 'SSA_index'
ff_jitter = DF.Jitter == jitter
ff_stream = DF.stream == stream


filtered = DF.loc[ff_model & ff_jitter & ff_parameter & ff_stream, :].drop_duplicates(subset=['cellid', 'SI_source']) #good, no duplicates


# ToDo filter by activity levels.
# filtered  = filtered.loc[ff_activity,:]

pivoted = filtered.pivot(index='cellid', columns='SI_source', values='value')


# the names of colums in pivoted are: 'fir2x15_pred', 'fir2x15_resp', 'stp2_pred', 'stp2_resp'

pivoted['null_MES'] = (pivoted.fir2x15_resp - pivoted.fir2x15_pred)**2
pivoted['alt_MES'] = (pivoted.stp2_resp - pivoted.stp2_pred)**2
pivoted['MSE_delta'] = pivoted['alt_MES'] - pivoted['null_MES']

mseDF = pivoted.loc[:,('null_MES', 'alt_MES', 'MSE_delta')]
mseDF = mseDF.astype(float)

# Compare the MSE between models without STP and with STP
fig, ax = plt.subplots()

# scatterplot data
x = mseDF.null_MES
y = mseDF.alt_MES
color = 'purple'

# deltaplot data
z = mseDF.MSE_delta
w = z-z
SI_delta = np.array([w, z])
# calculated confidence intervals
CI = bs.ci(data=z, statfunction=np.mean, n_samples=10000, method='pi')
ranks = st.ranksums(SI_delta[0, :], SI_delta[1, :])

if which_plot == 'scatter':
    mseDF.plot('null_MES', 'alt_MES', kind='scatter', c=color, s=40, alpha=0.5, ax=ax, label='{}'.format(stream), picker=True)
    ax.set_ylim(ax.get_xlim())
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k-')
    pick_id = mseDF.index.tolist()

elif which_plot == 'deta':
    # pair delta plots
    ax.plot([0, 1], SI_delta, color='gray', alpha=0.6, marker='o')
    ax.plot([0, 1], [0,np.mean(z)], color=color, linewidth='4')

    ax.axhline(0, linestyle='--', color='black')
    ax.fill_between([0.9,1.1], CI[0], CI[1], color=color, alpha=0.5)
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0,1])
    ax.set_xticklabels(('Core model\n(baseline)', 'STP Model'))
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('Δ MSE', fontsize=15)

else:
    raise ValueError("which_plot value {}. It must be 'scatter' or 'delta'".format(which_plot))


def onpick(event):
    ind = event.ind
    for ii in ind:
        # try:
        print('plotting\nindex: {}, cellid: {}'.format(ii, pick_id[ii]))
        # ToDo take this away whene everything works

        for modelname in modelnames:
            op.cell_psth(pick_id[ii], modelname)

fig.canvas.mpl_connect('pick_event', onpick)
fig.suptitle('MSE of SI prediction\n{}  VS  {}'.format(modelnames[0], modelnames[1]), fontsize=20)

