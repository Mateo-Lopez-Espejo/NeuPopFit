import pandas as pd
import joblib as jl
import numpy as np
import oddball_DF as odf
import seaborn as sns
import os

''' correlates the STP parameter values to the SSA index, across streams'''



# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
tail = '180718_DF_only_env_only_jal_jackknife_3_architectures'
tail = '180803_DF_only_env_only_jal_jackknife_3_architectures'

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


filtered = DF.loc[ff_param & (ff_Jitter | ff_Jitterna) & (ff_resp | ff_respna) & ff_stream, :]
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

