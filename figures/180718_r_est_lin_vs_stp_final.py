import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sts
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os

#### ploting parameters
# this block for linear vs stp
modelname1 = 'odd1_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
shortname1 = 'Linear STRF prediction'
modelname2 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
shortname2 = 'STP STRF prediction'
suptitile = 'linear model vs STP model'

# this block for the stp vs wc-stp
# modelname1 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
# shortname1 = 'r_test STP'
# modelname2 = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
# shortname2 = 'r_test WC-STP'
# suptitile = 'STP model vs WC-STP model'

# to be aware, interactive plotting only works properly whenn plotting a single model
modelnames = [modelname1, modelname2]
color1 = '#FDBF76' # yellow for linear model
color2 = '#CD679A' # pink for stp model


parameter = 'r_test' # this script is designed to deal with single values per recording (ignoring other vars)

# stream = ['f1', 'f2', 'cell']
stream = ['cell'] # unused var

# Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
Jitter = ['Jitter_Both'] # unused var

# goodness of fit filter
metric = 'r_test'
threshold = 0

# activity level filter
# metric = 'activity'
# threshold = 0



######## script starts here

# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
tail = '180718_DF_only_env_only_jal_jackknife_3_architectures'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)


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

filtered = quality_filtered.loc[ff_param & ff_model, :]
more_parms =  ['cellid']
pivot_by = 'modelname'
values = 'value'

tidy = odf.make_tidy(filtered,pivot_by, more_parms, values)

# changes names of modelname for ese of interpretations
tidy = tidy.rename(columns={modelname1: shortname1,
              modelname2: shortname2})

pick_id = tidy.cellid.tolist()

# gets linear regression values for printing? plotting?
nonan = tidy.dropna()
x = nonan[shortname1]
y = nonan[shortname2]
linreg = sts.linregress(x, y)
print(linreg)

# lmplot (linearmodel plot) fuses FacetGrid and regplot. so fucking tidy!
# format passed to plt...
palette = ['black', '#D7D8D9'] # second color is gray for non significant points
line_kws = {'linestyle': '-'}
scatter_kws = {'alpha': 0.8,
               'color': 'black', # This is temporal until i realize how to calculate significance.
               'picker': True}

g = sns.lmplot(x=shortname1, y=shortname2,
               aspect =1, legend_out=False, palette=palette,
               fit_reg=False,
               line_kws=line_kws, scatter_kws=scatter_kws,
               ci=None, data=tidy)

fig = g.fig
ax = g.ax

# vertical an horizontal lines at 0
ax.axvline(0, color='black', linestyle='--') # vertical line at 0
ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0

# makes the plot more square, by making top ylim equal to right xlim
ax.set_ylim(bottom=-0.1, top=1.1)
ax.set_xlim(ax.get_ylim())

lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

ax.set_xlabel(shortname1, fontsize=20, color=color1)
ax.set_ylabel(shortname2, fontsize=20, color=color2)
ax.set_title('')
fig.suptitle(suptitile, fontsize=20)



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

