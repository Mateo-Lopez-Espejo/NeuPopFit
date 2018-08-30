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


parameters = ['SSA_index'] # right now only works with SSA_index

# stream = ['f1', 'f2', 'cell']
stream = ['cell']

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
resp_or_pred = 'resp'



######## script starts here
# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
# tail = '180806_DF_only_env_only_jal_jackknife_3_architectures_JK_SI_pval'
tail = '180813_DF_only_env_only_jal_jackknife_4_architectures_full_SI_pval'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)


DF = loaded.copy()
# takes the mean of jakkcnifed SI pvalues
# DF = odf.collapse_pvalues(DF)

# filter by goodnes of fit
quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold= threshold)
# filter by parameters
ff_param = quality_filtered.parameter.isin(parameters)
ff_model = quality_filtered.modelname.isin(modelnames)
ff_jitter = quality_filtered.Jitter.isin(Jitter)
ff_stream = quality_filtered.stream.isin(stream)
# ff_resppred = quality_filtered.resp_pred == resp_pred
# ff_badcell = ~quality_filtered.cellid.isin(['chn008b-c2']) # this is cherry picking

# filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream & ff_badcell, :]
filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream, :]

# makes into tidy form
more_parms =  ['modelname', 'cellid']# , 'Jitter', 'stream', ]
pivot_by = 'resp_pred'
values = 'value'
tidy = odf.make_tidy(filtered,pivot_by, more_parms, values)

# changes names of modelname for ease of interpretations
tidy = tidy.replace({modelname1: shortname1,
              modelname2: shortname2})


# calculates significance in difference between recorded SI and Predicted SI
tidy, sig_name, nsig_name = odf.tidy_significance(tidy,['resp', 'pred'],fn=odf.jackknifed_sign, alpha=0.05)

# pulls SI significance from the original DF, gives same shape as tidy and plugs it in
# filters like SSA_index, but usign pvals instead
ff_cellid = quality_filtered.cellid.isin(tidy.cellid.unique())
ff_param = quality_filtered.parameter == 'SI_pvalue'
ff_model = quality_filtered.modelname.isin(modelnames)
ff_jitter = quality_filtered.Jitter.isin(Jitter)
ff_stream = quality_filtered.stream.isin(stream)
filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream, :]

#pivots
more_parms =  ['modelname', 'cellid']
pivot_by = 'resp_pred'
values = 'value'
pvals = odf.make_tidy(filtered,pivot_by, more_parms, values)

# finds significants SI values
pvals['significant'] = pvals[resp_or_pred] < alpha

# prints info, formats for later plotting
sig_count = pvals.significant.sum()
nsig_count = pvals.shape[0] - sig_count
print('{}/{} cells with significant SI'.format(sig_count, pvals.shape[0], ))
sig_name = 'p<{} (n={})'.format(alpha, sig_count)
nsig_name = 'NS (n={})'.format(nsig_count)
pvals.significant.replace({True: sig_name, False: nsig_name}, inplace=True)

# adds significant column to tidy
tidy['significant'] = pvals.significant.values


# plugs in significance for the SI values


# checks the number of significant dots for each model
# mod1_sig = tidy.loc[(tidy.modelname == shortname1) & (tidy.significant == sig_name)].shape[0]
# mod1_nsig = tidy.loc[(tidy.modelname == shortname1) & (tidy.significant == nsig_name)].shape[0]
# mod2_sig = tidy.loc[(tidy.modelname == shortname2) & (tidy.significant == sig_name)].shape[0]
# mod2_nsig = tidy.loc[(tidy.modelname == shortname2) & (tidy.significant == nsig_name)].shape[0]
#
# print('Linear model: {}/{} significantly different'.format(mod1_sig, mod1_sig+mod1_nsig))
# print('STP model: {}/{} significantly different'.format(mod2_sig, mod2_sig+mod2_nsig))

pick_id = tidy.cellid.tolist()

# gets linear regression values for printing? plotting?
for architecture in tidy.modelname.unique():
    ff_architecture = tidy.modelname == architecture
    x = tidy.loc[ff_architecture, 'resp']
    y = tidy.loc[ff_architecture, 'pred']
    linreg = sst.linregress(x, y)
    print('{}: {}'.format(architecture,linreg))

# lmplot (linearmodel plot) fuses FacetGrid and regplot. so fucking r_tidy!
# format passed to plt...
palette = [color1, color2]
line_kws = {'linestyle': '-'}
scatter_kws = {'alpha': 0.8,
               'picker': True}

g = sns.lmplot(x='resp', y='pred', hue='modelname',  col='significant', #col='stream', row='Jitter',
               aspect =1, legend_out=False, palette=palette,
               line_kws=line_kws, scatter_kws=scatter_kws,
               ci=None, data=tidy)

fig = g.fig
axes = g.axes
axes = np.ravel(axes)

for ax in axes:
    # vertical an horizontal lines at 0
    ax.axvline(0, color='black', linestyle='--') # vertical line at 0
    ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0

    # makes the plot more square, by making top ylim equal to right xlim
    ax.set_ylim(bottom=ax.get_ylim()[0], top=ax.get_xlim()[1])

    # make the plot absolute square
    ax.set_xlim(left=-1, right=1)
    ax.set_ylim(ax.get_xlim())

    lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
    upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
    ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

    ax.set_xlabel('actual SI', fontsize=20)
    ax.set_ylabel('predicted SI', fontsize=20)
    # ax.set_title('')

    # adds format to the legend box
    legend = ax.get_legend()
    legend.set_title(None)
    legend.get_frame().set_linewidth(0.0)

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

