import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sts
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op

#### ploting parameters

# resp_pred = 'pred'
# modelname = 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

modelname1 = 'odd1_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
shortname1 = 'r_test linear'
modelname2 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal'
shortname2 = 'r_test STP'

suptitile = 'linear model vs STP model'

# to be aware, interactive plotting only works properly whenn plotting a single model
modelnames = [modelname1, modelname2]
color1 = 'C0'
color2 = 'C1'


parameter = 'r_test' # right now only works with SSA_index

# stream = ['f1', 'f2', 'cell']
stream = ['cell']

# Jitter = ['Jitter_Off', 'Jitter_On', 'Jitter_Both']
Jitter = ['Jitter_Both']

# goodness of fit filter
metric = 'r_test'
threshold = 0

# activity level filter
# metric = 'activity'
# threshold = 0



######## script starts here


loaded = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal_jackknife')

def stp_plot(parameter=parameter, modelnames=modelnames, Jitter=Jitter, stream=stream, threshold=threshold):
    DF = loaded.copy()
    DF = odf.collapse_jackknife(DF)

    # filter by goodnes of fit
    quality_filtered = odf.filter_by_metric(DF, metric=metric, threshold= threshold)

    # filter by parameters
    ff_param = quality_filtered.parameter == parameter
    ff_model = quality_filtered.modelname.isin(modelnames)
    ff_jitter = quality_filtered.Jitter.isin(Jitter)
    ff_stream = quality_filtered.stream.isin(stream)
    #ff_resppred = quality_filtered.resp_pred == resp_pred

    filtered = quality_filtered.loc[ff_param & ff_model, :]
    more_parms =  ['cellid']
    pivot_by = 'modelname'
    values = 'value'

    tidy = odf.make_tidy(filtered,pivot_by, more_parms, values)

    # changes names of modelname for ese of interpretations
    tidy = tidy.rename(columns={modelname1: shortname1,
                  modelname2: shortname2})

    pick_id = tidy.cellid.tolist()

    # gets linear regression values for printing? plotting?
    nonan = tidy.dropna()
    x = nonan[shortname1]
    y = nonan[shortname2]
    linreg = sts.linregress(x, y)
    print(linreg)

    # lmplot (linearmodel plot) fuses FacetGrid and regplot. so fucking tidy!
    # format passed to plt...
    palette = [color1, color2]
    line_kws = {'linestyle': '-'}
    scatter_kws = {'alpha': 0.8,
                   'picker': True}

    g = sns.lmplot(x=shortname1, y=shortname2,
                   aspect =1, legend_out=False, palette=palette,
                   line_kws=line_kws, scatter_kws=scatter_kws,
                   ci=None, data=tidy)

    fig = g.fig
    ax = g.ax

    # vertical an horizontal lines at 0
    ax.axvline(0, color='black', linestyle='--') # vertical line at 0
    ax.axhline(0, color='black', linestyle='--') # hortizontal line at 0

    # makes the plot more square, by making top ylim equal to right xlim
    ax.set_ylim(bottom=-0.1, top=ax.get_xlim()[1])
    ax.set_xlim(ax.get_ylim())

    lowerleft = np.max([np.min(ax.get_xlim()), np.min(ax.get_ylim())])
    upperright = np.min([np.max(ax.get_xlim()), np.max(ax.get_ylim())])
    ax.plot([lowerleft, upperright], [lowerleft, upperright], 'k--')

    ax.set_xlabel(shortname1)
    ax.set_ylabel(shortname2)
    ax.set_title('')
    fig.suptitle(suptitile)



    def onpick(event):
        ind = event.ind
        for ii in ind:
            for modelname in modelnames:
                try:
                    # print('{}'.format(pick_id[ii]))
                    print('plotting\nindex: {}, cellid: {}, modelname: {}'.format(ii, pick_id[ii], modelname))
                    op.cell_psth(pick_id[ii], modelname)
                except:
                    print('error plotting: index: {}, cellid: {}'.format(ii, pick_id[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)

    return g

g = stp_plot()
