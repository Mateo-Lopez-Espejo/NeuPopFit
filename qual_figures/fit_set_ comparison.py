import joblib as jl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scikits.bootstrap as bs

'''Figure 2. compares the performance of different models both in as single cell scatter plots and boxplots
alternatively barplots with standard error bars'''

# load original data frame with single cell values
DF = jl.load('C:\\Users\Mateo\PycharmProjects\qual_figures\\296\\171117_6model_all_eval_DF')

# tease out r_est from relevant STP model ('order' parameter)
parameter = ['r_est']
order = ['stp1pc first']

rest = DF.loc[(DF.parameter=='r_est') &
              (DF.order=='stp1pc first'), :].\
    drop(columns=['Jitter', 'act_pred', 'model_name','stream','order','parameter'])

# Organize in convenient columns
pivoted = rest.pivot(index='cellid', columns='paradigm', values='values')


# All boxplots
fig, ax = plt.subplots()
pivoted.boxplot(ax=ax)



# Scatter plots
fig, axes = plt.subplots(1,2)
fig.suptitle('model performance eval self vs eval novel')
axes[0].plot(axes[0].get_xlim(), axes[0].get_xlim(), color='black', linestyle='-', linewidth ='4', alpha=0.5 )
on_color = 'green'
off_color = 'purple'

# fit off, eval off(self) and on(novel)
off_self = pivoted.loc[:,['fit: Off, eval: Off']].as_matrix().squeeze()
off_novel = pivoted.loc[:,['fit: Off, eval: On']].as_matrix().squeeze()
axes[0].scatter(off_self, off_novel, s=35, color=off_color)

# fit on, eval on(self) and off(novel)
on_self = pivoted.loc[:,['fit: On, eval: On']].as_matrix().squeeze()
on_novel = pivoted.loc[:,['fit: On, eval: Off']].as_matrix().squeeze()
axes[0].scatter(on_self, on_novel, s=35, color=on_color)

# formating
fig.suptitle('fitted with jittered tones')
axes[0].set_xlabel('self')
axes[0].set_ylabel('novel')

# paired lines plot
# prepare the data
on_diff = on_novel-on_self
on_norm = np.zeros([2,46])
on_norm[0,:] = on_diff
on_mean = np.mean(on_diff)

off_diff = off_novel-off_self
off_norm = np.zeros([2,46])
off_norm[1,:] = off_diff
off_mean = np.mean(off_diff)

# plot
axes[1].plot(on_norm, color='gray', marker='o', alpha=0.6)
axes[1].plot([on_mean,0], color=on_color, linewidth='4')

axes[1].plot([1,2], off_norm, color='gray', marker='o', alpha=0.6)
axes[1].plot([1,2], [0,off_mean], color=off_color, linewidth='4')

# add boxplots
#axes[1].boxplot([on_diff, off_diff], notch=True, positions=[-1,3])

st.ranksums(on_diff,off_diff)

# add bootstraped confidence intervals
off_CI = bs.ci(data=off_diff, statfunction=np.mean, method='pi')
on_CI = bs.ci(data=on_diff, statfunction=np.mean, method='pi')

axes[1].fill_between([-0.5,0.5], on_CI[0], on_CI[1], color=on_color, alpha=0.5)
axes[1].fill_between([1.5, 2.5], off_CI[0], off_CI[1], color=off_color, alpha=0.5)
axes[1].axhline(0, linestyle='--', color='black')
