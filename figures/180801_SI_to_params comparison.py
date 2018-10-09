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



# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with reweighted channels
tail = '180813_DF_only_env_only_jal_jackknife_4_architectures_full_SI_pval'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

DF = loaded.copy()

DF = odf.collapse_jackknife(DF)
DF = odf.filter_by_metric(DF,threshold=0.2)

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

filtered = DF.loc[ff_param & (ff_Jitter | ff_Jitterna) & (ff_resp | ff_respna) & ff_stream & ff_model & ff_signif, :]
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
    # fig, ax = plt.subplots()
    # ax.scatter(x, y)
    print('stream: {}, parameter: {}, r={}, pvalue={}'.format(stream, STP_parm, linreg.rvalue, linreg.pvalue))

# now pool by stream

for STP_parm in  tidy.STP_parm.unique():
    to_regress = tidy.loc[(tidy.STP_parm == STP_parm), ['SSA_index', 'stp_val']]
    x = to_regress['SSA_index']
    y = to_regress['stp_val']
    # g = sns.regplot(x, y)
    linreg = sst.linregress(x, y)
    print('stream: pooled, parameter: {}, r={}, pvalue={:.3E}'.format(STP_parm, linreg.rvalue, Decimal(linreg.pvalue)))

g = sns.lmplot(x='SSA_index', y='stp_val', data=tidy, hue='STP_parm')