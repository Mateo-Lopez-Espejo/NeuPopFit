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

modelname = 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
parameter = 'tau'
#Jitter = 'Jitter_On'
stream = 'f1'



######## script starts here

def stp_plot(parameter=parameter, stream=stream, modelname=modelname):
    old = jl.load('/home/mateo/batch_296/171115_all_subset_fit_eval_combinations_DF')
    old = oddball_DF.update_old_format(old)

    new = jl.load('/home/mateo/oddball_analysis/pickles/180709_DF_all_parms_only_jal')
    cellids = old.cellid.unique().tolist()

    DFS = {'old': old, 'new': new}

    fig, ax = plt.subplots()


    df_list = list()

    for name, df in DFS.items():

        df = df.copy()
        # defines filters
        ff_model = df.modelname == modelname
        ff_param = df.parameter == parameter
        #ff_jitter = df.Jitter == Jitter
        #ff_actpred = df.act_pred == 'pred'
        ff_cell  = df.cellid.isin(cellids)

        ff_stream = df.stream == stream

        # filter the DFs
        df_filt = df.loc[ff_model & ff_param & ff_stream & ff_cell, ['cellid', 'value']]

        df_filt = oddball_DF.collapse_jackknife(df_filt)

        # check and eliminate duplicates
        print(df_filt.duplicated(['cellid']).any())
        df_filt = df_filt.drop_duplicates(subset=['cellid'])

        # set the indexes
        df_filt.set_index('cellid', inplace=True)

        # rename value columns
        df_filt = df_filt.rename(columns={'value': '{}_value'.format(name)})

        df_list.append(df_filt)

    DF = pd.concat(df_list, axis=1, sort=True)
    DF = DF.astype(float)
    DF = DF.dropna()

    pick_id = DF.index.tolist()


    # plotting
    DF.plot('old_value', 'new_value', kind='scatter', ax=ax, label='{}'.format(stream), picker=True)

    lims = np.asarray([ax.get_xlim(), ax.get_ylim()])
    ax.plot( ax.get_ylim(),  ax.get_ylim(), ls="--", c=".3")
    ax.legend()
    ax.set_title('{}'.format(parameter))
    fig.suptitle('stp params old fit vs new fit')

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
