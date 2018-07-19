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


modelname1 = 'odd1_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
modelname2 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'

modelnames = [modelname1, modelname2]
colors = ['C0', 'C1']

parameter = 'SSA_index'
stream = 'cell'

# populatio quality filter
metric = 'r_test'
#metric = 'activity'
threshold = 0.2

# filters for plotting SSA index
Jitter = 'Jitter_Both'
resp_pred = 'pred'

# eyeballed outliers

eyeball = odf.eyeball_outliers()


######## script starts here

loaded = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal_jackknife')

def stp_plot(parameter=parameter, stream=stream, modelnames=modelnames, threshold=threshold):
    DF = loaded.copy()
    DF = odf.collapse_jackknife(DF)

    # filter by goodnes of fit
    quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold=threshold)

    fig, ax = plt.subplots()

    for color, modelname in zip(colors, modelnames):
        modelname = modelname2
        color = 'C1'

        # filter by parameters
        ff_param = quality_filtered.parameter == parameter
        ff_model = quality_filtered.modelname == modelname
        ff_jitter = quality_filtered.Jitter == Jitter
        ff_stream = quality_filtered.stream == stream
        ff_resppred = quality_filtered.resp_pred == resp_pred
        ff_eyeball = ~quality_filtered.cellid.isin(eyeball)

        filtered = quality_filtered.loc[ff_param & ff_model & ff_jitter & ff_stream, :]

        # check and eliminate duplicates
        print('duplicates: {}'.format(filtered.duplicated(['cellid', 'resp_pred']).any()))
        filtered = filtered.drop_duplicates(subset=['cellid', 'resp_pred'])

        # pivots by resp pred
        pivot = filtered.pivot(index='cellid', columns='resp_pred', values='value')

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
