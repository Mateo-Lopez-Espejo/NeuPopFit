import joblib as jl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pandas as pd
import copy
import scikits.bootstrap as bs
''' FIgure 1. atempts to compare the performance of models with and without STP parameters.
Models fitted using oddball stimuli ad vocalizations
'''


################ loading pickles #####################
# This DF contains all values like activity, SI etc.
oddDF = jl.load('C:\\Users\Mateo\PycharmProjects\qual_figures\\296\\171115_all_subset_fit_eval_combinations_DF')
# This DF contains only r_values for different models, thus does not contain individual cells
#DF = jl.load('C:\\Users\Mateo\PycharmProjects\qual_figures\\296\\171118_act_prec_rval_6_model_DF')

# selecting data from oddball paradimg
jitter = ['On', 'Off'] # jitter for r_est is NAN
modelnames = ['env100e_fir20_fit01_ssa', 'env100e_stp1pc_fir20_fit01_ssa']
parameter = ['r_est']
stream = [''] # Stream for r_est is also NAN
cellids = ['gus016c-a2', 'gus016c-c1', 'gus016c-c2', 'gus016c-c3', 'gus019c-a1', 'gus019c-b1', 'gus019c-b2',
           'gus019c-b3', 'gus019c-b4', 'gus019d-b1', 'gus019d-b2', 'gus019e-a1', 'gus019e-b1', 'gus019e-b2',
           'gus020c-a1', 'gus020c-c1', 'gus020c-c2', 'gus020c-d1', 'gus021c-a1', 'gus021c-a2', 'gus021c-b1',
           'gus021c-b2', 'gus021f-a1', 'gus021f-a2', 'gus021f-a3', 'gus022b-a1', 'gus022b-a2', 'gus023e-c1',
           'gus023e-c2', 'gus023f-c1', 'gus023f-c2', 'gus023f-d1', 'gus023f-d2', 'gus025b-a1', 'gus025b-a2',
           'gus026c-a3', 'gus026d-a1', 'gus026d-a2', 'gus030d-b1', 'gus035a-a1', 'gus035a-a2', 'gus035b-c3',
           'gus036b-b1', 'gus036b-b2', 'gus036b-c1', 'gus036b-c2']

oddball_filtered = oddDF.loc[(oddDF.model_name.isin(modelnames)) &
                             (oddDF.parameter.isin(parameter)) &
                             (oddDF.cellid.isin(cellids)), :]
oddball_pivoted = oddball_filtered.pivot(index='cellid', columns='model_name', values='values')

#box plot with paired lines, it does not look as good as i expected
arr = oddball_pivoted.as_matrix()
difference = arr[:,1]-arr[:,0]
norm_arr = np.zeros([2,46])
norm_arr[1,:] = difference
trend = np.nanmean(difference)
fig, axes = plt.subplots(1,2)


axes[0].plot([0,1], norm_arr, color='gray', alpha=0.6, marker='o')
axes[0].plot([0,trend], color='C0', linewidth='4')
# calculated confidence intervals
diff_CI = bs.ci(data=difference, statfunction=np.mean, n_samples=10000, method='pi' )
ranks = st.ranksums(norm_arr[0,:], norm_arr[1,:])
print(ranks)
# draws confidence intervals
# axes[0].boxplot(difference, notch=True, positions=[2]) # breaks the plot for some reason
axes[0].axhline(0, linestyle='--', color='black')
axes[0].fill_between([0.5,1.5], diff_CI[0], diff_CI[1], color='C0', alpha=0.5)

# Scatter plot STP vs no STP
axes[1].plot(axes[1].get_xlim(), axes[1].get_xlim(), color='black', linestyle='-', linewidth ='4', alpha=0.5 )
axes[1].scatter(oddball_pivoted[modelnames[0]], oddball_pivoted[modelnames[1]], s=35)


# Using SSA index prediction as a proxi for model performance









####### Graveyard ##########

# shuffle the pairs, 10000 times, get the differences, plot the mean of the differences
# something is not working, mean differences from shuffled pairs is always the same
reps = 10000
mean_diff = list()
for i in range(reps):
    shuffled = copy.deepcopy(arr)
    np.random.shuffle(shuffled[0,:])
    np.random.shuffle(shuffled[1,:])
    diff_shuf = shuffled[1,:] - shuffled[0,:]
    shuf_mean = np.mean(diff_shuf)
    mean_diff.append(shuf_mean)

# this confidence intervals are overlaping. Variance within the groups overpower subtle
# differences due to addition of STP
STP_CI = bs.ci(data=arr[:,1], statfunction=np.mean, n_samples=10000, method='pi' )
noSTP_CI = bs.ci(data=arr[:,0], statfunction=np.mean, n_samples=10000, method='pi' )
# willcoxon is not better
st.ranksums(arr[:,0], arr[:,1])


