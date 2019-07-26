import pandas as pd
import joblib as jl
import numpy as np
import oddball_DF as odf
import seaborn as sns
import os
import scipy.stats as sst
import matplotlib.pyplot as plt
import itertools as itt
from decimal import Decimal


''' correlates the STP parameter values to the SSA index, across streams'''



# parameters
parameters = ['SSA_index', 'tau', 'u']

# stream = ['f1', 'f2', 'cell']
stream = ['f1', 'f2']

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

###############################################################################
# test files. the paths will be different between my desktop and laptop.
# pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
pickles = '/home/mateo/code/oddball_analysis/pickles'

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with reweighted channels
tail = '180813_DF_only_env_only_jal_jackknife_4_architectures_full_SI_pval'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

#########

# list of cells used to plot paper_figures/180730_final_combined.py model comparison
# only contains cells with significant real SI
goodcells = ['chn004b-a1', 'chn004c-b1', 'chn005d-a1', 'chn029d-a1',
       'chn062c-c1', 'chn062f-a2', 'chn063b-d1', 'chn063h-b1',
       'chn065c-d1', 'chn065d-c1', 'chn066b-c1', 'chn066c-a1',
       'chn067d-b1', 'chn073b-b1', 'eno001f-a1', 'eno002c-c1',
       'eno002c-c2', 'eno005d-a1', 'eno006d-c1', 'eno008e-b1',
       'eno013d-a1', 'eno035c-a1', 'gus016c-a1', 'gus016c-c2',
       'gus019d-b1', 'gus019e-a1', 'gus020c-a1', 'gus020c-c1',
       'gus021c-a1', 'gus021c-b1', 'gus021f-a2', 'gus023e-c1',
       'gus023f-c1', 'gus025b-a1', 'gus026d-a1', 'gus030d-b1',
       'gus035a-a1', 'gus035a-a2', 'gus036b-b1', 'gus036b-c1',
       'gus036b-c2', 'gus037d-a1', 'gus037d-a2', 'gus037e-d2']




######## deals with the real data
DF = loaded.copy()

DF = odf.collapse_jackknife(DF)
DF = odf.filter_by_metric(DF,threshold=0.2)


ff_cells = DF.cellid.isin(goodcells)
ff_param = DF.parameter.isin(['SSA_index', 'tau', 'u'])
ff_Jitter = DF.Jitter == 'Jitter_Both'
ff_Jitterna = pd.isna(DF.Jitter)
ff_resp = DF.resp_pred == 'resp'
ff_respna = pd.isna(DF.resp_pred)
ff_stream = DF.stream.isin(['f1', 'f2'])
ff_model = DF.modelname == 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'


# filters for cells with significant SI
sig_cell = DF.loc[(DF.parameter == 'SI_pvalue') & (DF.value >=0.05), :].cellid.unique()
ff_signif = DF.cellid.isin(sig_cell)

filtered = DF.loc[ff_cells & ff_param &
                  (ff_Jitter | ff_Jitterna) &
                  (ff_resp | ff_respna) &
                  ff_stream & ff_model & ff_signif, :]
filtered['to_pivot'] = filtered.parameter == 'SSA_index'
filtered.to_pivot.replace({True: 'SI', False: 'STP'}, inplace=True)


tidy = odf.make_tidy(filtered, pivot_by='parameter', more_parms=['modelname', 'cellid', 'stream'], values='value')

# drops NaN from the linear filter
tidy = tidy.dropna()

# stack tau and u on top, with their respective SI values, this duplicates SI values... sue me

tau = tidy.loc[:, ['modelname', 'cellid', 'stream', 'SSA_index', 'tau']]
tau['STP_parm'] = 'tau'
tau.rename(columns={'tau': 'stp_val'}, inplace=True)
u = tidy.loc[:, ['modelname', 'cellid', 'stream', 'SSA_index', 'u']]
u['STP_parm'] = 'u'
u.rename(columns={'u': 'stp_val'}, inplace=True)


tidy = pd.concat([tau,u], axis=0)

g = sns.lmplot(x='SSA_index', y='stp_val', hue='stream', col='STP_parm', row='modelname', data=tidy)

fig = g.fig
axes = g.axes
axes = np.ravel(axes)

for ax in axes:
    # vertical an horizontal lines at 0
    ax.axvline(0, color='black', linestyle='--') # vertical line at 0
    ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0


    # make the plot rectangular
    ax.set_xlim(left=-1, right=1)
    ax.set_ylim(bottom=-2.5, top=2.5)

    lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
    upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
    ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

    # ax.set_xlabel('actual SI', fontsize=20)
    # ax.set_ylabel('predicted SI', fontsize=20)
    # ax.set_title('')

    # # adds format to the legend box
    # legend = ax.get_legend()
    # legend.set_title(None)
    # legend.get_frame().set_linewidth(0.0)



fig.suptitle('SI correlates with STP parameters')

# calculates linear regression between stream SSA index and Tau for pooled streams for the
# reweighted STP STRF model

for stream, STP_parm in itt.product(tidy.stream.unique(), tidy.STP_parm.unique()):
    to_regress = tidy.loc[(tidy.STP_parm == STP_parm) & (tidy.stream == stream), ['SSA_index', 'stp_val']]
    x = to_regress['SSA_index']
    y = to_regress['stp_val']
    linreg = sst.linregress(x, y)
    print('stream: {}, parameter: {}, r={}, pvalue={}'.format(stream, STP_parm, linreg.rvalue, linreg.pvalue))

# now pool by stream
reg_dict = dict()
for STP_parm in  tidy.STP_parm.unique():
    to_regress = tidy.loc[(tidy.STP_parm == STP_parm), ['SSA_index', 'stp_val']]
    x = to_regress['SSA_index']
    y = to_regress['stp_val']
    linreg = sst.linregress(x, y)
    label = '{}, r={:.3f}, p={:.3E}'.format(STP_parm, linreg.rvalue, Decimal(linreg.pvalue))
    reg_dict[STP_parm] = label
    print('stream: pooled, parameter: {}, r={}, pvalue={:.3E}'.format(STP_parm, linreg.rvalue, Decimal(linreg.pvalue)))

# makes a copy of the array, and modifies the values of STP_parms to include the r and pvalues, horrible kludge.
to_plot = tidy.copy()
to_plot = to_plot.replace(reg_dict)

g = sns.lmplot(x='SSA_index', y='stp_val', data=to_plot, hue='STP_parm')
ax = g.ax

ax.axvline(0, color='black', linestyle='--') # vertical line at 0
ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0

# make the plot rectangular
ax.set_xlim(left=-1, right=1)
ax.set_ylim(bottom=-2.5, top=2.5)

# draws unity line
lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

