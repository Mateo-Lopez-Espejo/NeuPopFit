import joblib as jl
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import seaborn as sn

''' Remaking the oldes, jitter on vs jitter off and its effect on SSA. This should be model agnostic since i am 
comparing the calculated SI, not the SI from the predictions. That comes later'''

# older DF, wit the model without STP, it should not change anything
DF = jl.load('C:\\Users\Mateo\PycharmProjects\qual_figures\\296\\171115_all_subset_fit_eval_combinations_DF')

# newer DF only with models containing STP. has different estimation validation sets
#DF = jl.load('C:\\Users\Mateo\PycharmProjects\qual_figures\\296\\171117_6model_all_eval_DF')

ffparam = DF.parameter == 'SI'
ffact = DF.act_pred == 'actual'
ffchann = DF.stream == 'cell'

filtered = DF.loc[ffparam & ffact & ffchann, :].drop_duplicates(['cellid', 'Jitter'])
pivoted = filtered.pivot(index='cellid', columns='Jitter', values='values')



# starts plotting

x = pivoted.Off
y = pivoted.On
labelsize = 20

fig, axes = plt.subplots()
axes.scatter(x,y)
lims = [axes.get_xlim(), axes.get_ylim()]
lims =np.asarray([ll for ax in lims for ll in ax])
max = lims.max()
min = lims.min()
axes.set_xlim(min,max)
axes.set_ylim(min,max)
axes.plot([min,max], [min, max], 'k--')
sn.regplot(x,y)
axes.set_xlabel('SI (regular)', fontsize=labelsize)
axes.set_ylabel('SI (random)', fontsize=labelsize)
axes.tick_params(axis='both', labelsize=15)
axes.set_title('Effect of regularity\non SSA index', fontsize=labelsize)

# linear regression wiht Pearson's correlation included
linreg = stats.linregress(x,y)
print (linreg)
