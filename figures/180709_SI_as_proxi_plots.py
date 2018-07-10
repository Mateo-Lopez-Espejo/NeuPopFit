import joblib as jl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scikits.bootstrap as bs

'''Given the overfitting of the model and the prediction to silence, we can use the goodness of fit for the SSA index
as as proxi for the model perfomance'''

# Imports Old Df containting both the model w/o and with STP. Unfortunately, the model without STP was never done
# with the paradigm of fitting and evaluating with different subsets (randon vs regular intervals) of the trials.

DF = jl.load('/home/mateo/oddball_analysis/pickles/180709_DF_all_parms_only_jal')

#creates new column with model (2) and act_pred (2) combinations as a single variable
# TODO recache and replace act_pred with resp_pred
DF['SI_source'] = ['{}_{}'.format(model.split('_')[1], ap) for model, ap in zip(DF.modelname, DF.act_pred)]

# Filters for SSA index values, calculated for the whole cell (frequency independent) and, for the poled data of jitter on and of
ffjitter = DF.Jitter == 'Jitter_Both'
ffparameter = DF.parameter == 'SSA_index'
ffstream = DF.stream == 'cell'

filtered = DF.loc[ffjitter & ffparameter & ffstream, :].drop_duplicates(subset=['cellid', 'SI_source']) #good, no duplicates
# ToDo filter by activity levels.
# filtered  = filtered.loc[ff_activity,:]

pivoted = filtered.pivot(index='cellid', columns='SI_source', values='value')
# the names of colums in pivoted are: 'fir2x15_pred', 'fir2x15_resp', 'stp2_pred', 'stp2_resp'

pivoted['Core_MES'] = (pivoted.fir2x15_resp - pivoted.fir2x15_pred)**2
pivoted['STP_MES'] = (pivoted.stp2_resp - pivoted.stp2_pred)**2
pivoted['MSE_delta'] = pivoted['STP_MES'] - pivoted['Core_MES']

mseDF = pivoted.loc[:,('Core_MES', 'STP_MES', 'MSE_delta')]


# Compare the MSE between models without STP and with STP
fig, axes = plt.subplots(1,2)
ax1, ax2 = axes

#scatterplot
x = mseDF.Core_MES
y = mseDF.STP_MES
color = 'purple'

ax1.scatter(x, y, c=color, s=40, alpha=0.5)
ax1.set_ylim(ax1.get_xlim())
ax1.plot(ax1.get_xlim(), ax1.get_xlim(), 'k-')
ax1.set_ylabel('STP Model', fontsize=15)
ax1.set_xlabel('Core Model', fontsize=15)
ax1.tick_params(axis='both', labelsize=15)

# pair delta plots
z = mseDF.MSE_delta
w = z-z
SI_delta = np.array([w, z])



ax2.plot([0, 1], SI_delta, color='gray', alpha=0.6, marker='o')
ax2.plot([0, 1], [0,np.mean(z)], color=color, linewidth='4')


# calculated confidence intervals
CI = bs.ci(data=z, statfunction=np.mean, n_samples=10000, method='pi')
ranks = st.ranksums(SI_delta[0, :], SI_delta[1, :])

# rand_reg_ranks = st.ranksums(u,z)
#
# print('regular intervals CI')
# print(reg_ranks)
# print('random intervals CI')
# print(rand_ranks)
# print('rand vs reg wicoxon')
# print(rand_reg_ranks
#       )
# draws confidence intervals
# axes[0].boxplot(difference, notch=True, positions=[2]) # breaks the plot for some reason
ax2.axhline(0, linestyle='--', color='black')
ax2.fill_between([0.9,1.1], CI[0], CI[1], color=color, alpha=0.5)
ax2.set_xlim(-0.3, 1.3)
ax2.set_xticks([0,1])
ax2.set_xticklabels(('Core model\n(baseline)', 'STP Model'))
ax2.tick_params(axis='both', labelsize=15)
ax2.set_ylabel('Δ MSE', fontsize=15)

fig.suptitle('MSE of SI prediction', fontsize=20)

