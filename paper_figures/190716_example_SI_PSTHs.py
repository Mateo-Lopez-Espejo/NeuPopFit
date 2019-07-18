import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sst
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os

# this block for the linear vs wc-stp
modelname1 = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname1 = 'LN STRF'
modelname2 = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname2 = 'RW-STP STRF'


modelnames = [modelname1, modelname2]
shortnames = [shortname1, shortname2]


# this block specifies models for the barplot comparing all architectures
LN = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
LN_name = 'LN_STRF'
glob_STP = 'odd.1_fir.2x15-stp.2-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
glob_STP_name = 'global STP STRF'
loc_STP = 'odd.1_stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
loc_STP_name = 'local STP STRF'
RW_STP = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
RW_STP_name = 'RW STP STRF'

all_models = {LN_name: LN, glob_STP_name: glob_STP,
              loc_STP_name: loc_STP, RW_STP_name: RW_STP}



color1 = '#FDBF76'  # yellow for linear model
color2 = 'purple'
color3 = '#CD679A'  # pink for stp model
color4 = '#5054a5'

color1 = '#FDBF76'  # yellow for linear model
color2 = '#5054a5' # blue for CW STP
model_colors = [color1, color2]

# axes subtitles
subtitle1 = 'r_test comparison'
subtitle2 = 'SSA Index (SI): calculated from response vs prediction'

# parameters
parameters = ['jk_r_test', 'SSA_index', 'SI_pvalue']

# stream = ['f1', 'f2', 'cell']
stream = ['cell']

# Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0

# limit to force values to
lowerlimit = -0.2

# SI pvlaue
alpha = 0.05
pval_set = 'resp'


########################################################################################################################
######## script starts here
# test files. the paths will be different between my desktop and laptop.
# pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
pickles = '/home/mateo/code/oddball_analysis/pickles'

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
# tail = '180718_DF_only_env_only_jal_jackknife_3_architectures' # old fitting
# tail = '180803_DF_only_env_only_jal_jackknife_3_architectures'  # newer fitting
tail = '180813_DF_only_env_only_jal_jackknife_4_architectures_full_SI_pval'


filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

# prefilters DF relevant for both SSA_index and r_test

DF = loaded.copy()

ff_param = DF.parameter.isin(parameters)
ff_model = DF.modelname.isin(modelnames)
ff_jitter = DF.Jitter.isin(Jitter) | pd.isnull(DF.Jitter)
ff_stream = DF.stream.isin(stream) | pd.isnull(DF.stream)
ff_badcells = ~DF.cellid.isin(['chn019a-d1', 'chn022c-a1', 'chn019a-c1']) # this cells have a nan value for the response SI

pre_filtered = DF.loc[ff_param & ff_model & ff_jitter & ff_stream & ff_badcells, :]


########################################################################################################################
### SI
print(' \nright plot')
goodcells = odf.filter_by_metric(DF,metric=metric, threshold=threshold)
ff_goodcell = pre_filtered.cellid.isin(goodcells.cellid.unique())
ff_ssa = pre_filtered.parameter == 'SSA_index'
si_filt = pre_filtered.loc[ff_ssa & ff_goodcell, :]

# makes tidy
more_parms = ['modelname', 'cellid']
pivot_by = 'resp_pred'
values = 'value'
si_tidy = odf.make_tidy(si_filt, pivot_by, more_parms, values)

# changes names of modelname for ese of interpretations
si_tidy = si_tidy.replace({modelname1: shortname1,
                           modelname2: shortname2})

# mean of Jackknife
si_tidy = odf.collapse_jackknife(si_tidy, columns=['resp', 'pred'])

# values minor to -0.2 are forced to -0.2
vals = si_tidy.loc[:, ['resp','pred']].values
ff_outlier = vals < lowerlimit
vals[ff_outlier] = lowerlimit
si_tidy['resp'] = vals[:, 0]
si_tidy['pred'] = vals[:, 1]

# repeats tidy for SI pvaluefor resp and "cell" stream
ff_goodcell = pre_filtered.cellid.isin(goodcells.cellid.unique())
ff_ssa = pre_filtered.parameter == 'SI_pvalue'
pv_filt = pre_filtered.loc[ff_ssa & ff_goodcell, :]
# makes tidy
more_parms = ['modelname', 'cellid']
pivot_by = 'resp_pred'
values = 'value'
pv_tidy = odf.make_tidy(pv_filt, pivot_by, more_parms, values)
# changes names of modelnames
pv_tidy = pv_tidy.replace({modelname1: shortname1,
                           modelname2: shortname2})

# changes pvalues to significance boolean and changes name to include number of points complying
pv_tidy['significant'] = pv_tidy[pval_set] <= alpha
# since reponse is model independent, looks at SI significance in a single model
pv_single_mod = pv_tidy.loc[pv_tidy.modelname==shortname1, :]
sig_count = pv_single_mod.significant.sum()
nsig_count = (pv_single_mod.shape[0] - sig_count)
print('{}/{} cells with significant SI using shuffle test'.format(sig_count, pv_single_mod.shape[0]))
SI_significant_name = 'p<{} (n={})'.format(alpha, sig_count)
SI_Nsignificant_name = 'NS (n={})'.format(nsig_count)
SI_significances = [SI_significant_name, SI_Nsignificant_name]
pv_tidy.significant.replace({True: SI_significant_name, False: SI_Nsignificant_name}, inplace=True)

# renames columns
pv_tidy.rename(columns={'resp': 'resp_pval', 'pred': 'pred_pval'}, inplace=True)

# concatenates the pvalue column into the si_mse DF, uses indexes to make sure of proper alignment
si = si_tidy.set_index(['cellid', 'modelname'])
pv = pv_tidy.set_index(['cellid', 'modelname'])
concat = pd.concat([si,pv], axis=1)
si_toplot = concat.reset_index()

# drops rows with na
si_toplot =si_toplot.dropna()

# gets mean and correlation coeficient
for short in shortnames:
    ff_short = si_toplot.modelname==short
    ff_pval = si_toplot.significant==SI_significant_name

    wdf = si_toplot.loc[ff_short & ff_pval, :]
    resp = wdf['resp'].values
    pred = wdf['pred'].values
    linreg = sst.linregress(resp, pred)
    print('{}: resp mean {:.3f}, pred mean {:.3f}, corcoef {:.3f}, slope {:.3f}'.
          format(short, np.mean(resp), np.mean(pred), linreg.rvalue, linreg.slope))

########################################################################################################################
### plotting

# fig, si_ax = plt.subplots()
# for model, color in zip(shortnames, model_colors):
#     # plots regression lines independent of significance
#
#     ff_model = si_toplot.modelname == model
#     # full_reg = si_toplot.loc[ff_model]
#     # z = full_reg.resp
#     # w = full_reg.pred
#     # sns.regplot(z, w, ax=si_ax, color='black', scatter=False, ci=None)
#
#     ff_sig = si_toplot.significant == SI_significant_name
#     toplot = si_toplot.loc[ff_model & ff_sig]
#     x = toplot.resp
#     y = toplot.pred
#
#     sig_scat_kws = {'s':30}
#     lab = '{} {}'.format(model, SI_significant_name)
#     sns.regplot(x, y, ax=si_ax, color=color, marker='o', label=lab, ci=None,
#                 scatter_kws=sig_scat_kws)

# lmplot (linearmodel plot) fuses FacetGrid and regplot. so fucking r_tidy!
# format passed to plt...
toplot = si_toplot.loc[si_toplot.significant == SI_significant_name]

palette = {shortname1: color1,
           shortname2: color2}  # second color is gray for non significant points
line_kws = {'linestyle': '-'}
scatter_kws = {'alpha': 0.8,
               'picker': True}

g = sns.lmplot(x='resp', y='pred', hue='modelname',
               aspect=1, legend_out=False, palette=palette,
               fit_reg=True,
               line_kws=line_kws, scatter_kws=scatter_kws,
               ci=None, data=toplot)

fig = g.fig
si_ax = g.ax


# adds format
# vertical an horizontal lines at 0
si_ax.axvline(0, color='black', linestyle='--')  # vertical line at 0
si_ax.axhline(0, color='black', linestyle='--')  # hortizontal line at 0

# makes the plot more square, by making top ylim equal to right xlim...
si_ax.set_ylim(bottom=si_ax.get_ylim()[0], top=si_ax.get_xlim()[1])
# and by making the bottom ylim and the left xlim equal to lowerlimit
# make the plot absolute square
si_ax.set_xlim(left=lowerlimit)
si_ax.set_ylim(bottom=lowerlimit)

lowerleft = np.max([np.min(si_ax.get_xlim()), np.min(si_ax.get_ylim())])
upperright = np.min([np.max(si_ax.get_xlim()), np.max(si_ax.get_ylim())])
si_ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

si_ax.set_xlabel('actual SI', fontsize=20)
si_ax.set_ylabel('predicted SI', fontsize=20)
si_ax.set_title(subtitle2, fontsize=20)

# adds format to the legend box
legend = si_ax.legend(loc='upper left')
legend.set_title(None)
legend.get_frame().set_linewidth(0.0)



pick_id = toplot.cellid.tolist()


def onpick(event):
    ind = event.ind
    for ii in ind:
        for modelname in modelnames:
            try:
                # print('{}'.format(pick_id[ii]))
                print('\nplotting\nindex: {}, cellid: {}, modelname: {}'.format(ii, pick_id[ii], modelname))
                op.cell_psth(pick_id[ii], modelname)
            except:
                print('error plotting: index: {}, cellid: {}'.format(ii, pick_id[ii]))


fig.canvas.mpl_connect('pick_event', onpick)

example_cell = 'chn066b-c1'
op.cell_psth(example_cell, modelnames[0])
# op.cell_psth(example_cell, modelnames[1])

