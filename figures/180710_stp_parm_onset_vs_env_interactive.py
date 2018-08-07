import oddball_DF
import oddball_test as ot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
import oddball_post_procecing as opp
import oddball_plot as op

#### ploting parameters

# resp_pred = 'pred'
# modelname = 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

modelname1 = 'odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
modelname2 = 'odd1_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

modelnames = [modelname1, modelname2]


parameter = 'SSA_index'

stream = 'f1'

# filters for plotting SSA index
Jitter = 'Jitter_On'
resp_pred = 'resp'


######## script starts here

def stp_plot(parameter=parameter, stream=stream, modelnames=modelnames):
    old = jl.load('/home/mateo/batch_296/171115_all_subset_fit_eval_combinations_DF')
    old = oddball_DF.relevant_from_old_DF(old)


    # function deprecated. DF deleted. to recreated pull from modelnames :
    # ['odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
    #  'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
    #  'odd1_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
    #  'odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal']
    new = jl.load('/home/mateo/oddball_analysis/pickles/180710_DF_all_parms_all_load_only_jal')
    cellids = old.cellid.unique().tolist()

    DF = pd.concat([old, new], sort=True)

    fig, ax = plt.subplots()


    df_list = list()

    for ii, modelname in enumerate(modelnames):

        df = DF.copy()
        # defines filters
        ff_model = df.modelname == modelname
        ff_param = df.parameter == parameter
        ff_jitter = df.Jitter == Jitter
        ff_resppred = df.resp_pred == resp_pred
        ff_cell  = df.cellid.isin(cellids)

        ff_stream = df.stream == stream

        # filter the DFs
        if parameter == 'SSA_index':
            df_filt = df.loc[ff_resppred & ff_jitter & ff_model & ff_param & ff_stream & ff_cell, ['cellid', 'value']]
        elif parameter in ['tau', 'u']:
            df_filt = df.loc[ff_model & ff_param & ff_stream & ff_cell, ['cellid', 'value']]

        df_filt = oddball_DF.collapse_jackknife(df_filt)

        # check and eliminate duplicates
        print('duplicates: {}'.format(df_filt.duplicated(['cellid']).any()))
        df_filt = df_filt.drop_duplicates(subset=['cellid'])

        # set the indexes
        df_filt.set_index('cellid', inplace=True)

        # rename value columns

        df_filt = df_filt.rename(columns={'value': '{}'.format(modelname)})

        df_list.append(df_filt)

    DF = pd.concat(df_list, axis=1, sort=True)
    DF = DF.astype(float)
    DF = DF.dropna()

    pick_id = DF.index.tolist()


    # plotting
    DF.plot(modelnames[0], modelnames[1], kind='scatter', ax=ax, label='{}'.format(stream), picker=True)

    lims = np.asarray([ax.get_xlim(), ax.get_ylim()])
    ax.plot( ax.get_ylim(),  ax.get_ylim(), ls="--", c=".3")
    ax.legend()
    ax.set_title('{}'.format(parameter))
    fig.suptitle('stp params\n{}  VS  {}'.format(modelnames[0], modelnames[1]))

    def onpick(event):
        ind = event.ind
        for ii in ind:
            # try:
            print('plotting\nindex: {}, cellid: {}'.format(ii, pick_id[ii]))
            # ToDo take this away whene everything works
            modelname = 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
            op.cell_psth(pick_id[ii], modelname)
            # except:
            #     print('error plotting: index: {}, cellid: {}'.format(ii, pick_id[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)
    return DF

DF = stp_plot()
