import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sst
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os
import itertools as itt

LN = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
LN_name = 'LN_STRF'
glob_STP = 'odd.1_fir.2x15-stp.2-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
glob_STP_name = 'global STP STRF'
loc_STP = 'odd.1_stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
loc_STP_name = 'local STP STRF'
RW_STP = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
RW_STP_name = 'RW STP STRF'

all_models = {LN: LN_name, glob_STP: glob_STP_name,
              loc_STP: loc_STP_name, RW_STP: RW_STP_name}

color1 = '#FDBF76'  # yellow for linear model
color2 = 'purple'
color3 = '#CD679A'  # pink for stp model
color4 = 'royalblue'
model_colors = [color1, color2, color3, color4]

# barplot params
# left plot
left_ylabel = 'mean R value'
left_suptitle = 'model goodness of fit'

# right plot
right_ylabel = 'mean MSE'
right_suptitle = 'model SI value prediction error'

# text sieze
labelsize = 25
titlesize = 35
ticksize = 20


# parameters
parameters = ['r_test', 'SSA_index', 'SI_pvalue']
stream = ['cell']
Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0

# limit to force values to
lowerlimit = -0.2

# SI pvlaue thresholf for siginifcance
alpha = 0.05
pval_set = 'resp' # unused

# right plot alternatives: 1. Mean standard error of the population, 2. correlation coefficient of the population,
# 3. mean of the mean standard error for the jk SI of each individual unit.
alternative = 3

########################################################################################################################
######## script starts here
# test files. the paths will be different between my desktop and laptop.
pickles = '{}/pickles'.format(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
tail = '180813_DF_only_env_only_jal_jackknife_4_architectures_full_SI_pval'
filename = os.path.normcase('{}/{}'.format(pickles, tail))
loaded = jl.load(filename)

DF = loaded.copy()

ff_params = DF.parameter.isin(parameters)
ff_models = DF.modelname.isin(all_models.keys())
ff_jitter = DF.Jitter.isin(Jitter) | pd.isnull(DF.Jitter)
ff_stream = DF.stream.isin(stream) | pd.isnull(DF.stream)
ff_badcells = ~DF.cellid.isin(['chn019a-d1', 'chn022c-a1', 'chn019a-c1']) # this cells have a nan value for the response SI

pre_filtered = DF.loc[ff_params & ff_models & ff_jitter & ff_stream & ff_badcells, :]
pre_filtered.replace(all_models, inplace=True)

########################################################################################################################
### r_test

ff_param = pre_filtered.parameter == 'r_test'
r_filt = pre_filtered.loc[ff_param, ['cellid', 'modelname', 'value']]

# values are signle element arrays, transforms into floats, stores under a new column with convenient name for plotting
r_filt['value'] = np.stack(r_filt.value.values).squeeze() # long format, ready to plot as barplot

# makes tidy to calculate paired statistical test
r_tidy = odf.make_tidy(r_filt, pivot_by='modelname', more_parms=['cellid'], values='value')

############
### SI

# common procesign for all alternateives
significant_si = odf.filter_by_metric(pre_filtered, metric='SI_pvalue', threshold=alpha)
ff_param = significant_si.parameter == 'SSA_index'
si_filt = significant_si.loc[ff_param, ['cellid', 'modelname', 'resp_pred', 'value']]

# alternative 1: calculate the population MSE between response SI and prediction SI for each model
if alternative == 1:
    # SI values are arrays of jackknifed SI values. takes the mean of the jacknife values. TODO is there a better approach
    mse_col_name = 'MSE across the population'
    si_filt['value'] = np.mean(np.stack(si_filt.value.values), axis=1)

    # to compare between recorded and predicted SI pivots by resp_pred
    si_resp_pred = odf.make_tidy(si_filt, pivot_by='resp_pred', more_parms=['cellid', 'modelname'], values='value')

    # iterates over every model and calculates the mean squared error between actual and predicted SI
    model_SI_MSE = dict.fromkeys(all_models.values())
    for model in all_models.values():
        ff_singleMod = si_resp_pred.modelname == model
        mod_arr = si_resp_pred.loc[ff_singleMod, ['resp', 'pred']].values
        mse = np.mean((mod_arr[:,0] - mod_arr[:,1])**2)
        model_SI_MSE[model] = mse

# alternative 2: calculates the linear regression between the response SI and prediction SI for each model
elif alternative == 2:
    # todo, for this to be consisten with regression from other figure, filtering by SI significance is necesry.

    # SI values are arrays of jackknifed SI values. takes the mean of the jacknife values.
    mse_col_name = 'linear_regression across the population'
    si_filt['value'] = np.mean(np.stack(si_filt.value.values), axis=1)

    # to compare between recorded and predicted SI pivots by resp_pred
    si_resp_pred = odf.make_tidy(si_filt, pivot_by='resp_pred', more_parms=['cellid', 'modelname'], values='value')

    # iterates over every model and calculates the mean squared error between actual and predicted SI
    model_SI_linreg = dict.fromkeys(all_models.values())
    for model in all_models.values():
        ff_singleMod = si_resp_pred.modelname == model
        mod_arr = si_resp_pred.loc[ff_singleMod, ['resp', 'pred']].values
        linreg = sst.linregress(mod_arr[:,0], mod_arr[:,1])
        model_SI_linreg[model] = linreg

# alternatively 3: calculates the MSE between the jackknifed response SI and prediction SI for each cell, barplot the means
elif alternative == 3:
    mse_col_name = 'mean of cell SI MSE'
    si_mse = odf.make_tidy(si_filt, pivot_by='resp_pred', more_parms=['cellid', 'modelname'], values='value')

    # DF values are lists of 5 jackknife SI values,
    # organizes in a matrix of dimentions C x J x R where C is the number of cells, J number of jackknifes, and R resp and pred
    si_jk_vals = si_mse.loc[:, ['resp', 'pred']].values
    si_jk_vals = np.stack([np.stack(si_jk_vals[:, 0]), np.stack(si_jk_vals[:, 1])], axis=2)

    # calculates the MSE for each cell using the response vs predicted SI values
    mse_arr = np.mean((si_jk_vals[:,:,0] - si_jk_vals[:,:,1])**2, axis=1)
    si_mse[mse_col_name] = mse_arr
    si_mse = si_mse.loc[:, ['cellid', 'modelname', mse_col_name]]

    # alternatively claculates a lineare regression and returns r_value, works like shit
    # regression = np.zeros(shape=si_jk_vals.shape[0])
    # for ii in range(regression.size):
    #     _, _, reg, _, _ = sst.linregress(si_jk_vals[ii,:,:])
    #     regression[ii] = reg
    # si_mse[mse_col_name] = regression


    # pivot by modelname
    si_tidy = odf.make_tidy(si_mse, pivot_by='modelname', more_parms=['cellid'], values=mse_col_name)

    model_unit_mean_MSE = dict.fromkeys(all_models.values())
    for model in all_models.values():
        model_mse_mean = np.mean(si_tidy[model].values)
        model_unit_mean_MSE[model] = model_mse_mean

########################################################################################################################
## ploting

fig, axes = plt.subplots(1,2)
axes = np.ravel(axes)

########### r_test plot, left panel  ###########

r_plot_ax = axes[0]
r_plot = sns.barplot(x='modelname', y='value', data=r_filt, ci=None, ax=r_plot_ax, order=list(all_models.values()),
                     palette=model_colors)
si_plot = op.model_progression(x='modelname', y='value', data=r_filt, mean=False, ax=r_plot_ax,
                               order=list(all_models.values()), palette=model_colors, collapse_by=0)

# r_plot = sns.swarmplot(x='modelname', y='value', data=r_filt, ax=r_plot_ax, order=list(all_models.values()),
#                      palette=model_colors)

# adds statistical significance labels
# iterates over each consecutive column pair
col_list = list(all_models.values())
print('r_test comparison:\n')
for cc, col in enumerate(col_list[:-1]):
    # select two consecutive columns
    col1 = col
    col2 = col_list[cc+1]

    xx = r_tidy[col1].values
    yy = r_tidy[col2].values
    w_test = sst.wilcoxon(xx,yy)
    print('{} vs {}: {}'.format(col1, col2, w_test))

    # sets a key for significance
    pval = w_test.pvalue
    if pval > 0.05:
        sig_key = 'ns'
    elif pval <= 0.05 and pval > 0.01:
        sig_key = '*'
    elif pval <= 0.01 and pval > 0.0001:
        sig_key = '**'
    elif pval <= 0.0001:
        sig_key = '***'
    else: # this should never happen
        sig_key = ''

    y = np.mean(r_tidy[col2]) + 0.005
    h = 0.001

    x1, x2 = cc, cc+1
    r_plot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=3, color='k')
    r_plot.text((x1+x2)*.5, y+h, sig_key, ha='center', va='bottom', color='k', fontsize=15)


########### SI MSE plot, right panel  ###########

if alternative == 1: # population MSE
    si_plot_ax = axes[1]
    x_pos = range(4)
    hights = list(model_SI_MSE.values())
    si_plot = plt.bar(x=x_pos, height=hights, color=model_colors)

elif alternative == 2: # population linregress

    model_SI_linreg = {key: val.slope for key, val in model_SI_linreg.items()}
    si_plot_ax = axes[1]
    x_pos = range(4)
    hights = list(model_SI_linreg.values())
    si_plot = plt.bar(x=x_pos, height=hights, color=model_colors)

elif alternative == 3: # mean of individual unit MSE

    si_plot_ax = axes[1]
    si_plot = sns.barplot(x='modelname', y=mse_col_name, data=si_mse, ci=None, ax=si_plot_ax, order=list(all_models.values()),
                          palette=model_colors)
    si_plot = op.model_progression(x='modelname', y=mse_col_name, data=si_mse, mean=False, ax=si_plot_ax,
                                   order= list(all_models.values()), palette=model_colors, collapse_by=0)

    # si_plot = sns.swarmplot(x='modelname', y=mse_col_name, data=si_mse, ax=si_plot_ax, order=list(all_models.values()),
    #                       palette=model_colors)

    # adds statistical significance labels
    # iterates over each consecutive column pair
    col_list = list(all_models.values())
    print('\nSI-MSE comparison:\n')
    for cc, col in enumerate(col_list[:-1]):
        # select two consecutive columns
        col1 = col
        col2 = col_list[cc+1]

        xx = si_tidy[col1].values
        yy = si_tidy[col2].values
        w_test = sst.wilcoxon(xx,yy)
        print('{} vs {}: {}'.format(col1, col2, w_test))

        # sets a key for significance
        pval = w_test.pvalue
        if pval > 0.05:
            sig_key = 'ns'
        elif pval <= 0.05 and pval > 0.01:
            sig_key = '*'
        elif pval <= 0.01 and pval > 0.0001:
            sig_key = '**'
        elif pval <= 0.0001:
            sig_key = '***'
        else: # this should never happen
            sig_key = ''

        y = np.mean(si_tidy[col1]) + 0.005
        h = 0.001

        x1, x2 = cc, cc+1
        si_plot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=3, color='k')
        si_plot.text((x1+x2)*.5, y+h, sig_key, ha='center', va='bottom', color='k', fontsize=15)

### adds format to the axes

labelsize = 25
titlesize = 35
ticksize = 20
modelnamesize = 15

r_plot.set_ylabel(left_ylabel, fontsize=labelsize)
r_plot.set_xlabel('model architecture', fontsize=labelsize)
r_plot.set_title(left_suptitle, fontsize=titlesize)
r_plot.tick_params(axis='x', which='major', labelsize=modelnamesize)
r_plot.tick_params(axis='y', which='major', labelsize=ticksize)

si_plot.set_ylabel(right_ylabel, fontsize=labelsize)
si_plot.set_xlabel('model architecture', fontsize=labelsize)
si_plot.set_title(right_suptitle, fontsize=titlesize)
si_plot.tick_params(axis='x', which='major', labelsize=modelnamesize)
si_plot.tick_params(axis='y', which='major', labelsize=ticksize)












































