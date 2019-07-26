import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sst
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os

'''
add a on pick function to plot specific neuron examples from the scatter plot

Works with older versions of NEMS (githash: 3a25cc5259f30e2b7a961e4a9fac2477e57b8144)
and nems_db (githash: 3fefdb537b100c346486266c97f18e3f55cb5086)

'''


#### ploting parameters
# this block for linear vs stp
modelname1 = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname1 = 'Linear STRF prediction'
modelname2 = 'odd.1_stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname2 = 'STP STRF prediction'
suptitile = 'linear model vs STP model'


# to be aware, interactive plotting only works properly whenn plotting a single model
modelnames = [modelname1, modelname2]
shortnames = [shortname1, shortname2]
color1 = '#FDBF76'  # yellow for linear model
color2 = '#CD679A'  # pink for stp model

parameters = ['r_test',
              'se_test']  # this script is designed to deal with single values per recording (ignoring other vars)

# stream = ['f1', 'f2', 'cell']
stream = ['cell']  # unused var

# Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
Jitter = ['Jitter_Both']  # unused var

# goodness of fit filter
metric = 'r_test'
threshold = 0

# activity level filter
# metric = 'activity'
# threshold = 0


######## script starts here

# test files. the paths will be different between my desktop and laptop.
# pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
pickles = '/home/mateo/code/oddball_analysis/pickles'

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
# tail = '180718_DF_only_env_only_jal_jackknife_3_architectures' # old fitting
tail = '180803_DF_only_env_only_jal_jackknife_3_architectures'  # newer fitting

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

DF = loaded.copy()
# DF = odf.collapse_jackknife(DF)

# filter by goodnes of fit
quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold=threshold)

# filter by parameters
ff_param = quality_filtered.parameter.isin(parameters)
ff_model = quality_filtered.modelname.isin(modelnames)
ff_jitter = quality_filtered.Jitter.isin(Jitter)
ff_stream = quality_filtered.stream.isin(stream)
# ff_resppred = quality_filtered.resp_pred == resp_pred

filtered = quality_filtered.loc[ff_param & ff_model, :]

# makes into r_tidy format holding only necesary parameters e.g. cellid, and pivoting by one the parameter
# to be ploted as x and y
more_parms = ['cellid', 'parameter']
pivot_by = 'modelname'
values = 'value'

tidy = odf.make_tidy(filtered, pivot_by, more_parms, values)

# changes names of modelname for ese of interpretations
tidy = tidy.rename(columns={modelname1: shortname1,
                            modelname2: shortname2})

# finds significance between columns and generate a new parameter
# tidy, sig_name, nsig_name = odf.tidy_significance(tidy,shortnames,fn=odf.jackknifed_sign, alpha=0.05)

# finds significance using the precalculated mean and standard error values.
wdf = filtered.replace({modelname1: shortname1, modelname2: shortname2})
wdf = wdf.set_index(['modelname', 'parameter', 'cellid'])
print(wdf.index.duplicated().any())
wdf = pd.DataFrame(index=wdf.index, data=wdf.value.astype(np.float))
wdf = wdf.unstack(['modelname', 'parameter'])
wdf.columns = wdf.columns.droplevel(0)

# calculates significance
wdf['significant', 'r_test'] = (np.abs(wdf[shortname1, 'r_test'] - wdf[shortname2, 'r_test']) >
                                (wdf[shortname1, 'se_test'] + wdf[shortname2, 'se_test']))

# drop standard errors
wdf = wdf.xs('r_test', axis=1, level='parameter')

# renames significant level to keep old format
sig_count = wdf.significant.sum()
nsig_count = wdf.shape[0] - sig_count
print('{}/{} significant'.format(sig_count, wdf.shape[0]))
alpha = 0.05
sig_name = 'p<{} (n={})'.format(alpha, sig_count)
nsig_name = 'NS (n={})'.format(nsig_count)
wdf = wdf.replace({True: sig_name, False: nsig_name})

tidy = wdf

# gets linear regression values for printing? plotting?
nonan = tidy.dropna()
x = nonan[shortname1]
y = nonan[shortname2]
linreg = sst.linregress(x, y)
print(linreg)

# lmplot (linearmodel plot) fuses FacetGrid and regplot. so fucking r_tidy!
# format passed to plt...
palette = {sig_name: 'black',
           nsig_name: '#D7D8D9'}  # second color is gray for non significant points
line_kws = {'linestyle': '-'}
scatter_kws = {'alpha': 0.8,
               # 'color': 'black', # This is temporal until i realize how to calculate significance.
               'picker': True}

g = sns.lmplot(x=shortname1, y=shortname2, hue='significant',
               aspect=1, legend_out=False, palette=palette,
               fit_reg=False,
               line_kws=line_kws, scatter_kws=scatter_kws,
               ci=None, data=tidy)

fig = g.fig
ax = g.ax


# finds the example cell
if 1:
    example_cell = 'chn066b-c1'
    ii = tidy.index.get_loc(example_cell)
    excell = tidy.iloc[ii]
    ax.scatter(excell[shortname1], excell[shortname2], color='red')

# vertical an horizontal lines at 0
ax.axvline(0, color='black', linestyle='--')  # vertical line at 0
ax.axhline(0, color='black', linestyle='--')  # hortizontal line at 0

# makes the plot more square, by making top ylim equal to right xlim
ax.set_ylim(bottom=-0.1, top=1.1)
ax.set_xlim(ax.get_ylim())

lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

ax.set_xlabel(shortname1, fontsize=20, color=color1)
ax.set_ylabel(shortname2, fontsize=20, color=color2)
ax.set_title(suptitile, fontsize=20)
# fig.suptitle(suptitile, fontsize=20)

# adds format to the legend box
legend = ax.get_legend()
legend.set_title(None)
legend.get_frame().set_linewidth(0.0)

plt.tight_layout()

# holds pickid for interactive plot
pick_id = tidy.cellid.tolist()


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
