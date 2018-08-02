import joblib as jl
import numpy as np
import pandas as pd
import os
import oddball_DF as odf
import seaborn as sns

'''
compares the SI value calcualted for the cell as in Ulanovsky, vs the mean of the SI value for each of the channels
this to test the claim in page 13 of the manuscript
'''


# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# this load also contains onset fits
# tail = '180710_DF_all_parms_all_load_only_jal_jackknife'

# this load only contain envelope fits but includesthe STP with channel crosstalk
tail = '180718_DF_only_env_only_jal_jackknife_3_architectures'

filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

DF = loaded.copy()

ff_parameter = DF.parameter == 'SSA_index'
ff_resp = DF.resp_pred == 'resp'
ff_jitter = DF.Jitter == 'Jitter_Both'
filtered = DF.loc[ff_parameter & ff_resp & ff_jitter, :]

tidy = odf.make_tidy(filtered, pivot_by='stream', more_parms=['cellid', 'modelname'], values='value')

# organizes full jackknife values and gets significance measurement

def row_mean(row):
    f1 = np.asarray(row['f1'])
    f2 = np.asarray(row['f2'])
    stacked = np.stack([f1,f2], axis=0)
    meaned = np.mean(stacked, axis=0)
    return meaned

tidy['mean'] = tidy.apply(row_mean, axis=1)
# removes nans
tidy = tidy.dropna()

sig_DF,_,_ = odf.tidy_significance(tidy,['cell', 'mean'], fn=odf.jackknifed_sign,alpha=0.05)
g = sns.lmplot('cell', 'mean', sig_DF, hue='significant')

fig = g.fig
ax = g.ax

ax.set_xlabel('Ulanovsky cell SI calculation')
ax.set_ylabel('mean of SI calculated for each frequency')
ax.axvline(0, color='black', linestyle='--') # vertical line at 0
ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0

# make the plot absolute square
ax.set_xlim(left=-1, right=1)
ax.set_ylim(ax.get_xlim())

lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')
