import oddball_functions as of
import oddball_DB as odb
import matplotlib.pyplot as plt
import numpy as np
import scikits.bootstrap as bts
import nems.signal as signal
import itertools as itt
import pandas as pd
import nems.modelspec as ms
import nems.recording as recording


# helper functions
def draw_single_envelope(pre_silence_duration, stim_duration, amplitude, post_silence_duration, freq_sampling, channel,
                         shape_fn=None):
    '''

    :param pre_silence_duration: float, time in seconds
    :param stim_duration: float, time in seconds
    :param amplitude: float. 0.17782 correspond to 60 sbSPL... i think
    :param post_silence_duration: float, time in seconds
    :param freq_sampling: int time bins per second
    :param shape_fn: fucntion defined from time(x) in seconds, to amplitude(y) e.g. exponential decay
    :return: a 1d ndarray defining the stimulus envelope
    '''
    # generates the stim wave form
    if shape_fn == None:
        bin_number = np.ceil(stim_duration * freq_sampling).astype(int)
        stim = np.full(bin_number, amplitude, dtype=np.float)
    else:
        raise NotImplementedError('poke mateo to do it')

    pre_stim_silence = np.zeros(np.ceil(pre_silence_duration * freq_sampling).astype(int))
    post_stim_silence = np.zeros(np.ceil(post_silence_duration * freq_sampling).astype(int))

    single_chan = np.concatenate((pre_stim_silence, stim, post_stim_silence), axis=0)

    full_stim = np.zeros((2, single_chan.shape[0]))
    full_stim[channel, :] = single_chan

    return full_stim


def synth_TiledSignal(**arguments):
    fs = arguments['freq_sampling']
    amp = arguments['amplitude']

    stim_num = 10

    stim_dict = dict.fromkeys([str(ii) for ii in range(stim_num)])
    # define the first long steady state stimulus
    stim_dict['0'] = draw_single_envelope(pre_silence_duration=1, stim_duration=1, amplitude=amp,
                                          post_silence_duration=0, channel=0, freq_sampling=fs)

    # defines the second stimulus, shorter one, with short wait after
    stim_dict['1'] = draw_single_envelope(pre_silence_duration=2, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=0, freq_sampling=fs)

    # defines the third stimulus, equal to previous one. with longer wait after

    stim_dict['2'] = draw_single_envelope(pre_silence_duration=0.3, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=0, freq_sampling=fs)

    # defines the third stimulus, equal to previous one, but in the other channel

    stim_dict['3'] = draw_single_envelope(pre_silence_duration=2, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=1, freq_sampling=fs)

    # defines the third stimulus, equal to previous one, in the original channel to se interaction

    stim_dict['4'] = draw_single_envelope(pre_silence_duration=0.3, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=0, freq_sampling=fs)


    # repeats the same with the other channel

    stim_dict['5'] = draw_single_envelope(pre_silence_duration=1, stim_duration=1, amplitude=amp,
                                          post_silence_duration=0, channel=1, freq_sampling=fs)

    # defines the second stimulus, shorter one, with short wait after
    stim_dict['6'] = draw_single_envelope(pre_silence_duration=2, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=1, freq_sampling=fs)

    # defines the third stimulus, equal to previous one. with longer wait after

    stim_dict['7'] = draw_single_envelope(pre_silence_duration=0.3, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=1, freq_sampling=fs)

    # defines the third stimulus, equal to previous one, but in the other channel

    stim_dict['8'] = draw_single_envelope(pre_silence_duration=2, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=0, channel=0, freq_sampling=fs)

    # defines the third stimulus, equal to previous one, in the original channel to se interaction

    stim_dict['9'] = draw_single_envelope(pre_silence_duration=0.3, stim_duration=0.1, amplitude=amp,
                                          post_silence_duration=1, channel=1, freq_sampling=fs)



    # generates two channel stimuli form the single channel envelopes
    # for (tone_name, tone),(ff, freq) in itt.product(tone_dict.items(), enumerate(frequencies)):
    #     bins = tone.shape[0]
    #     stim = np.zeros((2,bins))
    #     stim[ff,:] = tone
    #     stim_name = '{}_{}'.format(freq, tone_name)
    #     stim_dict[stim_name] = stim

    # generates the epochs, for now it concatenates all the epochs with no spaces in between, the silences are defined
    # by the waveforms from first, second, third ...
    current_time = 0
    df = list()
    for stim_name, stim in stim_dict.items():
        dur = stim.shape[1] / fs
        start = current_time
        end = current_time + dur
        d = {'start': start,
             'end': end,
             'name': stim_name}
        df.append(d)
        current_time = end

    df = pd.DataFrame(df)
    # orders columns as standard, only aesthetic
    df = df[['start', 'end', 'name']]

    tiledsingal = signal.TiledSignal(fs=fs, data=stim_dict, name='sim_sig', recording='sim_rec', epochs=df)

    return tiledsingal


def predict_synth(modelspecs, **arguments):
    synth_sig = synth_TiledSignal(**arguments)
    signals = {'stim': synth_sig}
    synth_rec = recording.Recording(signals)
    # iterates over each modelspec from  jackknifed predictions
    synth_preds = [ms.evaluate(synth_rec, m) for m in modelspecs]

    return synth_preds


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
        signal = ctx['val'][0][key]

        # folds by oddball epochs
        folded_sig = of.extract_signal_oddball_epochs(signal, sub_epoch, super_epoch)

        resp_pred[key] = folded_sig

    # calculate confidence intervals, organizes in a dictionary of equal structure as the matrix
    conf_dict = {outerkey: {innerkey:
                                np.asarray([bts.ci(innerval[:, 0, tt], np.mean, n_samples=100, method='pi')
                                            for tt in range(innerval.shape[2])])
                            for innerkey, innerval in outerval.items()} for outerkey, outerval in resp_pred.items()}

    fig, axes = plt.subplots(1, 2, sharey=True)

    axes = np.ravel(axes)

    for ax, RP in zip(axes, resp_pred.keys()):

        for color, freq in zip(colors, frequencies):

            for linestyle, rate in zip(linestyles, rates):
                outerkey = RP
                innerkey = freq + '_' + rate

                matrix = resp_pred[outerkey][innerkey]
                psth = np.nanmean(matrix, axis=0).squeeze()
                conf_int = conf_dict[outerkey][innerkey]
                onset = (psth.shape[0] / 3) * period
                offset = (psth.shape[0] * 2 / 3) * period

                t = np.arange(0, psth.shape[0] * period, period)

                ax.plot(t, psth, color=color, linestyle=linestyle, label=innerkey)
                ax.fill_between(t, conf_int[:, 0], conf_int[:, 1], color=color, alpha=0.2)

                ax.axvline(onset, color='black')
                ax.axvline(offset, color='black')

                ax.set_ylabel('spike rate (Hz)')
                ax.legend(loc='upper left', fontsize='xx-small')

        ax.set_title(outerkey)

    fig.suptitle('{} {}'.format(meta['cellid'], meta['modelname']))

    return fig, axes


def cartoon(ctx):
    modelspecs = ctx['modelspecs']
    original_stim = ctx['rec']['stim']
    fs = original_stim.fs
    max_amp = np.max(original_stim.as_continuous())

    arguments = {'pre_silence_duration': None,
                 'stim_duration': None, 'amplitude': max_amp,
                 'post_silence_duration': None,
                 'freq_sampling': fs,
                 'shape_fn': None}

    synth_recs = predict_synth(modelspecs, **arguments)

    fig, ax = plt.subplots()
    stim = synth_recs[0]['stim'].as_continuous().T
    ax.plot(stim)
    for ii, rec in enumerate(synth_recs):
        synth_pred = rec['pred'].as_continuous().T + 1 + ii
        ax.plot(synth_pred)

    return fig, ax


# higher level interface

def cell_psth(cellid, modelname, batch=296):
    ctx = odb.load_single_ctx(cellid, batch, modelname)

    fig, axes = psth(ctx)

    return fig, axes


def cell_cartoon(cellid, modelname, batch=296):
    ctx = odb.load_single_ctx(cellid, batch, modelname)

    fig, axes = cartoon(ctx)

    return fig, axes
