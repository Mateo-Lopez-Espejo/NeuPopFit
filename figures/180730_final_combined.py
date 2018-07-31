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

modelnames = [modelname1, modelname2]
shortnames = [shortname1, shortname2]

color1 = '#FDBF76'  # yellow for linear model
color2 = '#CD679A'  # pink for stp model
model_colors = [color1, color2]

# axes subtitles
subtitle1 = 'r_test comparison'
subtitle2 = 'SSA Index (SI): calculated from response vs prediction'

# parameters
parameters = ['jk_r_test', 'SSA_index']

# stream = ['f1', 'f2', 'cell']
stream = ['cell']

# Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0.15

# limit to force values to
lowerlimit = -0.2

######## script starts here
# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
tail = '180718_DF_only_env_only_jal_jackknife_3_architectures'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

# prefilters DF relevant for both SSA_index and r_test

DF = loaded.copy()

# quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold= threshold)

ff_param = DF.parameter.isin(parameters)
ff_model = DF.modelname.isin(modelnames)
ff_jitter = DF.Jitter.isin(Jitter) | pd.isnull(DF.Jitter)
ff_stream = DF.stream.isin(stream) | pd.isnull(DF.stream)

pre_filtered = DF.loc[ff_param & ff_model & ff_jitter & ff_stream, :]

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
    print('{} mean r_test: {:+.3f}'.format(short, np.mean(r_tidy[short])))


### SI
print(' \nright plot')
goodcells = odf.filter_by_metric(DF, threshold=0.2)
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

# drops columns with na
si_tidy = si_tidy.dropna()

# values minor to -0.2 are forced to -0.2
vals = si_tidy.loc[:, ['resp','pred']].values
ff_outlier = vals < lowerlimit
vals[ff_outlier] = lowerlimit
si_tidy['resp'] = vals[:, 0]
si_tidy['pred'] = vals[:, 1]

# gets mean and correlation coeficient
for short in shortnames:
    wdf = si_tidy.loc[si_tidy.modelname==short, :]
    resp = wdf['resp'].values
    pred = wdf['pred'].values
    linreg = sst.linregress(resp, pred)
    print('{}: resp mean {:+.3f}, pred mean {:+.3f}, corcoef {:+.3f}'.format(short, np.mean(resp), np.mean(pred), linreg.rvalue))



### plotting
fig, axes = plt.subplots(1, 2)
axes = np.ravel(axes)

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

    r_ax.scatter(x, y, color=color, label=sig)

# adds format
# vertical an horizontal lines at 0
r_ax.axvline(0, color='black', linestyle='--')  # vertical line at 0
r_ax.axhline(0, color='black', linestyle='--')  # hortizontal line at 0

# makes the plot more square, by making top ylim equal to right xlim
r_ax.set_ylim(bottom=-0.1, top=1.1)
r_ax.set_xlim(r_ax.get_ylim())

# ads identity line
lowerleft = np.max([np.min(r_ax.get_xlim()), np.min(r_ax.get_ylim())])
upperright = np.min([np.max(r_ax.get_xlim()), np.max(r_ax.get_ylim())])
r_ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

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
    ff_model = si_tidy.modelname == model
    toplot = si_tidy.loc[ff_model]
    x = toplot.resp
    y = toplot.pred
    #si_ax.scatter(x, y, color=color, label=model)
    sns.regplot(x, y, ax=si_ax, color=color, label=model, ci=None)

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
