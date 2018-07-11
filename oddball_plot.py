import oddball_functions as of
import oddball_db as odb
import matplotlib.pyplot as plt
import numpy as np
import scikits.bootstrap as bts
import nems.signal as signal


# base plottign functions

def my_bootstrap(data):
    # Bootstrap for mean confidence intervals
    # imput data as a list or 1d array of values
    # output the 95% confidence interval
    # based on scikyt.bootstrap.ci() .

    n_samples = 200  # number of samples
    alpha = 0.1  # two tailed alpha value, 90% confidence interval
    alpha = np.array([alpha / 2, 1 - alpha / 2])
    ardata = np.array(data)
    bootindexes = [np.random.randint(ardata.shape[0], size=ardata.shape[0]) for _ in
                   range(n_samples)]
    stat = np.array([np.nanmean(ardata[indexes]) for indexes in bootindexes])
    stat.sort(axis=0)
    nvals = np.round((n_samples - 1) * alpha)
    nvals = np.nan_to_num(nvals).astype('int')
    return stat[nvals]


def psth(ctx, sub_epoch=None, super_epoch=None):

    # todo pull the fs from somewhere
    fs = 100
    period = 1 / fs

    meta = ctx['modelspecs'][0][0]['meta']


    # organizes data

    resp_pred = dict.fromkeys(['resp', 'pred'])

    # color represents sound frequency
    colors = ['green', 'red']
    frequencies = ['f1', 'f2']

    # linestyle indicates preentation rate
    linestyles = ['-', ':']
    rates = ['std', 'dev']

    # pull the signals from the validation recording in ctx
    for key in resp_pred.keys():
        signal =  ctx['val'][0][key]

        # folds by oddball epochs
        folded_sig = of.extract_signal_oddball_epochs(signal, sub_epoch, super_epoch)

        resp_pred[key] =  folded_sig

    # calculate confidence intervals, organizes in a dictionary of equal structure as the matrix
    conf_dict = {outerkey : {innerkey :
                 np.asarray([bts.ci(innerval[:,0,tt], np.mean, n_samples=100, method='pi')
                             for tt in range(innerval.shape[2])])
                 for innerkey, innerval in outerval.items()} for outerkey, outerval in resp_pred.items()}



    fig, axes  = plt.subplots(1,2, sharey=True)

    axes = np.ravel(axes)

    for ax, RP in zip(axes, resp_pred.keys()):

        for color, freq in zip(colors, frequencies):

            for linestyle, rate in zip(linestyles, rates):

                outerkey = RP
                innerkey = freq + '_' + rate

                matrix = resp_pred[outerkey][innerkey]
                psth = np.nanmean(matrix,axis=0).squeeze()
                conf_int = conf_dict[outerkey][innerkey]
                onset = (psth.shape[0]/3) * period
                offset = (psth.shape[0] * 2/3 ) * period

                t = np.arange(0, psth.shape[0] * period, period)

                ax.plot(t, psth, color=color, linestyle=linestyle, label=innerkey)
                ax.fill_between(t, conf_int[:,0], conf_int[:,1], color=color, alpha=0.2)

                ax.axvline(onset , color='black')
                ax.axvline(offset , color='black')

                ax.set_ylabel('spike rate (Hz)')
                ax.legend(loc='upper left', fontsize='xx-small')

        ax.set_title(outerkey)

    fig.suptitle('{} {}'.format(meta['cellid'], meta['modelname']))

    return fig, axes


def simulated_stim(recording):
    # todo generates a snipped of stim,  to use as a cartoon and see the cell predicted response in a better way

    array  = np.array()
    fs = 100
    name = 'cartoon_stim'
    recording = recording

    signal = signal.RasterizedSignal.from_3darray(fs, array, name, recording, epoch_name='TRIAL',
                     chans=None, meta=None, safety_cheks=True)
    return signal



# higher level interface

def cell_psth(cellid, modelname, batch=296):

    ctx = odb.load_single_ctx(cellid, batch, modelname)

    fig, axes = psth(ctx)

    return fig, axes






