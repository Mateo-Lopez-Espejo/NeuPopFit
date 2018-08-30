import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sst
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os

#### ploting parameters

# this block for act vs pred SI across linear and STP models
modelname1 = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname1 = 'Linear model'
modelname2 = 'odd.1_stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname2 = 'STP model'

# this block for the stp vs wc-stp
# modelname1 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
# shortname1 = 'STP model'
# modelname2 = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
# shortname2 = 'WC-STP model'


# to be aware, interactive plotting only works properly whenn plotting a single model
modelnames = [modelname1, modelname2]
color1 = '#FDBF76' # yellow for linear model
color2 = '#CD679A' # pink for stp model


parameters = ['SSA_index', 'SI_pvalue'] # right now only works with SSA_index

stream = ['f1', 'f2', 'cell']
# stream = ['cell']

# Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter'NS (n=30)'_Both']
Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0.2

# activity level filter
# metric = 'activity'
# threshold = 0

# SI significance
alpha = 0.05
pval_stream = 'cell'



######## script starts here
# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
tail = '180806_DF_only_env_only_jal_jackknife_3_architectures_JK_SI_pval'
tail = '180813_DF_only_env_only_jal_jackknife_4_architectures_full_SI_pval'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)


DF = loaded.copy()
DF = odf.collapse_jackknife(DF,fn=np.nanmean)
DF[(DF==float("inf")) | (DF==float("-inf"))] = np.nan
DF = DF.dropna(subset=['value'])

# filter by goodnes of fit
quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold= threshold)

# compares to compare SI value against pvalue
ff_param = quality_filtered.parameter.isin(parameters)
ff_model = quality_filtered.modelname.isin(modelnames)
ff_jitter = quality_filtered.Jitter.isin(Jitter)
ff_stream = quality_filtered.stream.isin(stream)
ff_resp_pred = quality_filtered.resp_pred == 'resp'
filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream, :]
# makes into tidy form
more_parms = ['modelname', 'cellid', 'resp_pred', 'stream' ]
pivot_by = 'parameter'
values = 'value'
tidy_SI_vs_pval = odf.make_tidy(filtered, pivot_by, more_parms, values)


# renames the models fore readability
tidy_SI_vs_pval = tidy_SI_vs_pval.replace({modelname1: shortname1,
                            modelname2: shortname2})

lm = sns.lmplot(x='SSA_index', y='SI_pvalue', hue='resp_pred', col='stream', row='modelname', data=tidy_SI_vs_pval)


# filter by parameters
tidy_dict = dict.fromkeys(parameters)
filtered_dict = dict.fromkeys(parameters)
for param in parameters:
    ff_param = quality_filtered.parameter == (param)
    ff_model = quality_filtered.modelname.isin(modelnames)
    ff_jitter = quality_filtered.Jitter.isin(Jitter)
    ff_stream = quality_filtered.stream.isin(stream)
    ff_resp_pred = quality_filtered.resp_pred == 'resp'
    filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream, :]
    filtered_dict[param] = filtered

    # makes into tidy form
    more_parms =  ['modelname', 'cellid', 'resp_pred',]
    pivot_by = 'stream'
    values = 'value'
    tidy = odf.make_tidy(filtered,pivot_by, more_parms, values)

    # changes names of modelname for ease of interpretations
    tidy = tidy.replace({modelname1: shortname1,
                  modelname2: shortname2})

    tidy_dict[param] = tidy

tidy = tidy_dict['SSA_index']

SI_pval = tidy_dict['SI_pvalue']
SI_pval['SI_pvalue'] = SI_pval[pval_stream]

# finds significants SI values
SI_pval['SI_pvalue'] = SI_pval['SI_pvalue'] < alpha

# prints info, formats for later plotting
sig_count = SI_pval.SI_pvalue.sum()
nsig_count = SI_pval.shape[0] - sig_count
print('{}/{} cells with significant SI'.format(sig_count, SI_pval.shape[0], ))
sig_name = 'p<{} (n={})'.format(alpha, sig_count)
nsig_name = 'NS (n={})'.format(nsig_count)
SI_pval.SI_pvalue.replace({True: sig_name, False: nsig_name}, inplace=True)


tidy['SI_pvalue'] = SI_pval['SI_pvalue']

# renames the models fore readability
tidy = tidy.rename(columns={modelname1: shortname1,
                            modelname2: shortname2})




pick_id = tidy.cellid.tolist()

# lmplot (linearmodel plot) fuses FacetGrid and regplot. so fucking tidy!
# format passed to plt...
palette = [color1, color2]
palette = ['black', 'gray']
# line_kws = {'linestyle': '-'}
scatter_kws = {'alpha': 0.8,
               'picker': True}


g = sns.lmplot(x='f1', y='f2', hue='SI_pvalue', col='modelname', row='resp_pred',
               aspect =1, legend_out=False, palette=palette, scatter_kws=scatter_kws,
               fit_reg=False, x_jitter=0.01, y_jitter=0.01,
               ci=None, data=tidy)

fig = g.fig
axes = g.axes
axes = np.ravel(axes)

for ax in axes:
    # vertical an horizontal lines at 0
    ax.axvline(0, color='black', linestyle='--') # vertical line at 0
    ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0

    # make the plot absolute square
    ax.set_xlim(left=-0.5, right=0.5)
    ax.set_ylim(ax.get_xlim())

    lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
    upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
    ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

    # ax.set_xlabel('actual SI', fontsize=20)
    # ax.set_ylabel('predicted SI', fontsize=20)
    # # ax.set_title('')
    #
    # # adds format to the legend box
    # legend = ax.get_legend()
    # legend.set_title(None)
    # legend.get_frame().set_linewidth(0.0)

plt.tight_layout()


def onpick(event):
    ind = event.ind
    for ii in ind:
        for modelname in modelnames:
            try:
                # print('{}'.format(pick_id[ii]))
                print('plotting\nindex: {}, cellid: {}, modelname: {}'.format(ii, pick_id[ii], modelname))
                op.cell_psth(pick_id[ii], modelname)
            except:
                print('error plotting: index: {}, cellid: {}'.format(ii, pick_id[ii]))

fig.canvas.mpl_connect('pick_event', onpick)

