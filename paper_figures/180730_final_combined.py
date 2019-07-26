import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sst
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os
import pathlib as pl


"""
Works with older versions of NEMS (githash: 3a25cc5259f30e2b7a961e4a9fac2477e57b8144)
and nems_db (githash: 3fefdb537b100c346486266c97f18e3f55cb5086)
"""

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
# makes two versions of a r_tidy DF, one for each parameter
### r_test
print(' \nleft plot')
r_filt = pre_filtered.loc[pre_filtered.parameter == 'jk_r_test']
more_parms = ['cellid']
pivot_by = 'modelname'
values = 'value'
r_tidy = odf.make_tidy(r_filt, pivot_by, more_parms, values)
# changes names of modelname for ese of interpretations
r_tidy = r_tidy.rename(columns={modelname1: shortname1,
                                modelname2: shortname2})
# finds significance between columns and generate a new parameter
r_tidy, sig_name, nsig_name = odf.tidy_significance(r_tidy, shortnames, fn=odf.jackknifed_sign, alpha=0.05)

# gets mean and standard deviation for each model r_test
for short in shortnames:
    print('{} mean r_test: {:.3f}'.format(short, np.mean(r_tidy[short])))



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
fig, axes = plt.subplots(1, 2)
axes = np.ravel(axes)
markersize = 50
linewidth = 3


# first ax compares r_test between models, color code indicate significance
r_ax = axes[0]
for sig in r_tidy.significant.unique():
    if sig[0:2] == 'NS':
        # non significant, set color gray
        color = '#D7D8D9'
    elif sig[0:2] == 'p<':
        color = 'black'

    # filter the subset of the data
    toplot = r_tidy.loc[r_tidy.significant == sig, :]
    x = toplot[shortname1]
    y = toplot[shortname2]

    r_ax.scatter(x, y, color=color, label=sig, s=markersize)

    #example cell
    if 1:
        example_cell = 'chn066b-c1'
        excell = r_tidy.loc[(r_tidy.significant == sig) & (r_tidy.cellid == example_cell), [shortname1, shortname2]]
        r_ax.scatter(excell[shortname1], excell[shortname2], color= color, marker='p')

# adds format
# vertical an horizontal lines at 0
r_ax.axvline(0, color='black', linestyle='--', linewidth=linewidth)  # vertical line at 0
r_ax.axhline(0, color='black', linestyle='--', linewidth=linewidth)  # hortizontal line at 0

# makes the plot more square, by making top ylim equal to right xlim
r_ax.set_ylim(bottom=-0.1, top=1.1)
r_ax.set_xlim(r_ax.get_ylim())

# ads identity line
lowerleft = np.max([np.min(r_ax.get_xlim()), np.min(r_ax.get_ylim())])
upperright = np.min([np.max(r_ax.get_xlim()), np.max(r_ax.get_ylim())])
r_ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--', linewidth=linewidth)

# color and size for axis labels
r_ax.set_xlabel(shortname1, fontsize=20, color=color1)
r_ax.set_ylabel(shortname2, fontsize=20, color=color2)
r_ax.set_title(subtitle1, fontsize=20)
# fig.suptitle(suptitile, fontsize=20)

# adds format to the legend box
legend = r_ax.legend(loc='upper left')
legend.set_title(None)
legend.get_frame().set_linewidth(0.0)

# Second ax compares actual and predicted SI, different colors indicate different models
si_ax = axes[1]

for model, color in zip(shortnames, model_colors):
    # plots regression lines independent of significance

    ff_model = si_toplot.modelname == model
    # full_reg = si_toplot.loc[ff_model]
    # z = full_reg.resp
    # w = full_reg.pred
    # sns.regplot(z, w, ax=si_ax, color='black', scatter=False, ci=None)

    ff_sig = si_toplot.significant == SI_significant_name
    toplot = si_toplot.loc[ff_model & ff_sig]
    x = toplot.resp
    y = toplot.pred

    lab = '{} {}'.format(model, SI_significant_name)
    sns.regplot(x, y, ax=si_ax, color=color, marker='o', label=lab, ci=None,
                scatter_kws={'s':markersize}, line_kws={'linewidth':linewidth})

# adds format
# vertical an horizontal lines at 0
si_ax.axvline(0, color='black', linestyle='--', linewidth=linewidth)  # vertical line at 0
si_ax.axhline(0, color='black', linestyle='--', linewidth=linewidth)  # hortizontal line at 0

# makes the plot more square, by making top ylim equal to right xlim...
si_ax.set_ylim(bottom=si_ax.get_ylim()[0], top=si_ax.get_xlim()[1])
# and by making the bottom ylim and the left xlim equal to lowerlimit
# make the plot absolute square
si_ax.set_xlim(left=lowerlimit)
si_ax.set_ylim(bottom=lowerlimit)

lowerleft = np.max([np.min(si_ax.get_xlim()), np.min(si_ax.get_ylim())])
upperright = np.min([np.max(si_ax.get_xlim()), np.max(si_ax.get_ylim())])
si_ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--', linewidth=linewidth)

si_ax.set_xlabel('actual SI', fontsize=20)
si_ax.set_ylabel('predicted SI', fontsize=20)
si_ax.set_title(subtitle2, fontsize=20)

# adds format to the legend box
legend = si_ax.legend(loc='upper left')
legend.set_title(None)
legend.get_frame().set_linewidth(0.0)


for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=20)
    ax.title.set_size(30)
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)

# set figure to full size in tenrec screen
fig.set_size_inches(19.2, 9.79)

#
# root = pl.Path(f'/home/mateo/Pictures/STP_paper')
# filename = f'LN_RW-STP'
# if not root.exists(): root.mkdir(parents=True, exist_ok=True)
#
# png = root.joinpath(filename).with_suffix('.png')
# fig.savefig(png, transparent=True, dpi=100)
#
# svg = root.joinpath(filename).with_suffix('.svg')
# fig.savefig(svg, transparent=True)