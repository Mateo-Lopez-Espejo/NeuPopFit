import oddball_test as ot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
import oddball_post_procecing as opp
import oddball_plot as op
import oddball_DF as odf
import scipy.stats as sts

#### ploting parameters

# resp_pred = 'pred'
# modelname = 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

modelname1 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
modelname2 = 'odd1_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

modelnames = [modelname1, modelname2]
colors = ['green', 'red']

parameter = 'SSA_index'
stream = 'cell'

# populatio quality filter
metric = 'r_test'
#metric = 'activity'
threshold = 0

# filters for plotting SSA index
Jitter = 'Jitter_Both'
resp_pred = 'pred'

# eyeballed outliers

eyeball = odf.eyeball_outliers()


######## script starts here

def stp_plot(parameter=parameter, stream=stream, modelnames=modelnames, threshold=threshold):
    old = jl.load('/home/mateo/batch_296/171115_all_subset_fit_eval_combinations_DF')
    old = odf.relevant_from_old_DF(old)

    new = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal')
    cellids = old.cellid.unique().tolist()

    DF = pd.concat([old, new], sort=True)

    fig, ax = plt.subplots()


    df_list = list()

    for color, modelname in zip(colors, modelnames):

        df = DF.copy()
        # filter by modelname
        ff_model = df.modelname == modelname
        df_filt = df.loc[ff_model,:]

        # filter by goodnes of fit
        df_filt = odf.filter_df_by_metric(df_filt, metric=metric, threshold= threshold)
        if df_filt.empty is True:
            print('no cells with this filter')
            continue

        # filter by parameters
        ff_param = df_filt.parameter == parameter
        ff_jitter = df_filt.Jitter == Jitter
        ff_resppred = df_filt.resp_pred == resp_pred
        ff_stream = df_filt.stream == stream

        ff_eyeball = ~df_filt.cellid.isin(eyeball)


        if parameter == 'SSA_index':
            df_filt = df_filt.loc[ff_jitter & ff_param & ff_stream & ff_eyeball, :]
        elif parameter in ['tau', 'u']:
            raise NotImplementedError('tau or u not yet implemented')
            # df_filt = df.loc[ff_model & ff_param & ff_stream & ff_cell, ['cellid', 'value']]

        df_filt = odf.collapse_jackknife(df_filt)

        # check and eliminate duplicates
        print('duplicates: {}'.format(df_filt.duplicated(['cellid', 'resp_pred']).any()))
        df_filt.drop_duplicates(subset=['cellid', 'resp_pred'], inplace=True)

        # set the indexes
        # df_filt.set_index('cellid', inplace=True)  #take it off for pivot
        # rename value columns
        # df_filt = df_filt.rename(columns={'value': '{}'.format(modelname)})

        pivot = df_filt.pivot(index='cellid', columns='resp_pred', values='value')
        # get regresion metrics line ploting
        linreg = sts.linregress(pivot['resp'], pivot['pred'])
        slope = linreg.slope
        intercept = linreg.intercept
        rvalue = linreg.rvalue
        pivot.plot('resp', 'pred', kind='scatter', color=color,  ax=ax, label='{}'.format(modelname), picker=True)

        x = np.asarray(ax.get_xlim())
        ax.plot(x, intercept + slope * x, color=color, label='slope: {}, r_value: {} '.format(slope, rvalue))

        ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
        ax.legend()
        ax.set_title('{}'.format(parameter))
        pick_id = pivot.index.tolist()

        # rename columns
        pivot = pivot.rename(columns={'resp': 'resp@{}'.format(modelname),
                              'pred': 'pred@{}'.format(modelname)})
        df_list.append(pivot)

    DF = pd.concat(df_list, axis=1, sort=True)
    DF = DF.astype(float)
    DF = DF.dropna()

    # plotting
    def onpick(event):
        ind = event.ind
        for ii in ind:
            for modelname in modelnames:
                try:
                    print('plotting\nindex: {}, cellid: {}, modelname: {}'.format(ii, pick_id[ii], modelname))
                    # print(pick_id[ii])
                    op.cell_psth(pick_id[ii], modelname)
                except:
                    print('error plotting: index: {}, cellid: {}'.format(ii, pick_id[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)


    return DF

DF = stp_plot()
