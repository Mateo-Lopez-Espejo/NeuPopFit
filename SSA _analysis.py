
import numpy as np
import scipy.special
import scipy.stats as stats
import nems.preprocessing as pps
import warnings
import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal
import copy
import joblib as jl
import matplotlib.pyplot as plt


def set_oddball_epochs(signal):
    '''
    rename the signal epochs to a form generalized for oddball stimulus.
    i.e. (onset, standard, deviant) * (frequency 1, frequency 6).
    Adds a sound epoch in between the PreStimSilence and PostStimSilence

    Parameters
    ----------
    signal: a signal object

    Returns
    -------
    copy of the signal with renamed epochs and new sound epoch
    '''

    # regular expression to match either SubStim or STIM tags inlcuding sound frequency and rate or ONSET
    # e.g. "STIM_25000+ONSET" or "STIM_17500+0.05" or "SUBSTIM_17500+0.05"
    # todo check if the new Oddball implementation tags are being paresed capitalized or not (ask Stephen)
    regexp = r"((SubStim)|(STIM))_\d*\+((0\.\d*)|ONSET)"

    epoch_names_to_extract = ep.epoch_names_matching(signal.epochs, regexp)

    # defines deviant standard and onset for each sound frequency,
    # "STIM_25000+ONSET"
    #  ^1   ^2    ^3     1: prefix, 2: center_frequencie, 3: standard_deviant_rates
    center_frequencies = {float(name.split('_')[1].split('+')[0]) for name in epoch_names_to_extract}
    standard_deviant_rates = {name.split('_')[1].split('+')[1] for name in epoch_names_to_extract}
    standard_deviant_rates.remove('ONSET')
    standard_deviant_rates = {float(rate) for rate in standard_deviant_rates}
    prefix = epoch_names_to_extract[0].split('_')[0] # STIM or PreStim

    # Checks the input signal has the adecuates frequencies and rates dimentions
    if (len(standard_deviant_rates) != 2) or (len(center_frequencies) != 2):
        raise ValueError("epochs contain {} presetnations rates, 2 are required\n"
                         "epochs contain {} center frequencies, 2 are required".format(len(standard_deviant_rates),
                                                                                       len(center_frequencies)))

    # explicit statement of dict key to event tag mapping
    key_mapping = {'{}_{:.0f}+{}'.format(prefix, min(center_frequencies), 'ONSET') : 'f1_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), max(standard_deviant_rates)) : 'f1_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), min(standard_deviant_rates)) : 'f1_dev',
                   '{}_{:.0f}+{}'.format(prefix, max(center_frequencies), 'ONSET') : 'f2_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), max(standard_deviant_rates)) : 'f2_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), min(standard_deviant_rates)) : 'f2_dev'}

    # creates new event dataframe with modified epoch names
    oddball_epochs = copy.deepcopy(signal.epochs)
    for oldkey, newkey in key_mapping.items():
        oddball_epochs.name.replace(oldkey, newkey, inplace=True)

    updated_signal = signal._modified_copy(signal._data, epochs=oddball_epochs)



    # generates new events containing the sound time
    new_event_name = 'Sound'
    # checks size congruence of  PreStimSilence and PostStimSilence
    PreStim = oddball_epochs.loc[oddball_epochs.name == 'PreStimSilence', ['start', 'end']].as_matrix()
    PostStim = oddball_epochs.loc[oddball_epochs.name == 'PostStimSilence', ['start', 'end']].as_matrix()
    if PreStim.shape == PostStim.shape:
        pass
    else:
        raise ValueError('there are not equal number of PreStimSilence and PostStimSielence epochs')

    sound_epoch_matrix = np.stack([PreStim[:,1], PostStim[:, 0]], axis=1)
    updated_signal.add_epoch(new_event_name, sound_epoch_matrix)

    return updated_signal

def fold_oddball_signal(signal, sub_epoch):
    #TODO is this function really necesary? I think is already implemented in signal.as_matrix or similar

    '''
    given a signal, usually from an oddball experiment, finds the epoch names corresponding to all different
    stimulus combination i.e. center frequency (frequency 1 , frequency 2) * rate of presentation (standard, deviant)
    folds and returs and ordered dictionary of matrices

    Parameters
    ----------
    signal : A signal object
    sub_epoch : None, str
        if none, returns the whole REFERENCE epoch, otherwise returns the epoch contained within REFERENCE
        it should be 'Sound', 'PreStimSilence' or 'PostStimSilence'

    Returns
    -------
    folded_signal : dict
        Dictionary containing (M x N) matrixes with values for each sound type, where M is the repetition and N is time

    '''

    signal = signal.rasterize()

    sound_types = ['f1_onset', 'f1_std', 'f1_dev', 'f2_onset', 'f2_std', 'f2_dev']

    if sub_epoch == None:
        folded_signal = {sound_type: signal.extract_epoch(sound_type) for sound_type in
                     sound_types}
    elif sub_epoch in signal.epochs.name.unique():
        folded_signal = {sound_type: signal.extract_epoch(sub_epoch, overlapping_epoch=sound_type) for sound_type in
                         sound_types}
    else:
        raise ValueError("sub_epoch name is not withing epochs")

    return folded_signal

def SSA_index_2(recording):

    return None

def SSA_index(recording, subset = 'resp', return_clasified_responses = False):
    '''
    Given the recording from an SSA object, returns the SSA index (SI) as defined by Ulanovsky et al (2003)


    Parameters
    ----------
    recording : A Recording object
        Generally the output of `model.evaluate(phi, data)`??
    subset : string ['resp' or 'pred']
        Name of the subset of data from which to calculate the SI,
        either the response 'resp' or prediction 'pred'

    Returns
    -------
    SSA_index_dict : dict
        Dictionary containing the SI values for each of the sound frequency channels and independent of frequency

    '''

    if subset not in ['resp', 'pred']:
        raise ValueError("subset has to be 'resp' or 'pred'")

    working_signal = recording.get_signal(subset)

    if not isinstance(working_signal, signal.RasterizedSignal):
        raise ValueError('{} is not a RasterizedSignal'.format(subset))

    if working_signal.shape[0] > 1:
        raise NotImplementedError('multi-channel signals not supported yet')

    # TODO implemetne oddball clasification as an independent function
    # gets the clasiffied responses

    # extract relevant epochs
    # regular expression to match either SubStim or STIM tags inlcuding sound frequency and rate or ONSET
    # e.g. "STIM_25000+ONSET" or "STIM_17500+0.05" or "SUBSTIM_17500+0.05"
    # todo check if the new Oddball implementation tags are being paresed capitalized or not (ask Stephen)
    regexp = r"((SubStim)|(STIM))_\d*\+((0\.\d*)|ONSET)"

    epoch_names_to_extract = ep.epoch_names_matching(working_signal.epochs, regexp)

    # defines deviant standard and onset for each sound frequency,
    # maps the epoch tags into dictionary with keys generalized for oddball stimuli
    center_frequencies = {float(name.split('_')[1].split('+')[0]) for name in epoch_names_to_extract}
    standard_deviant_rates = {name.split('_')[1].split('+')[1] for name in epoch_names_to_extract}
    standard_deviant_rates.remove('ONSET')
    standard_deviant_rates = {float(rate) for rate in standard_deviant_rates}

    prefix = epoch_names_to_extract[0].split('_')[0] # STIM or PreStim

    # explicit statement of dict key to event tag mapping
    sound_types = {'f1_onset': '{}_{:.0f}+{}'.format(prefix, min(center_frequencies), 'ONSET'),
                   'f1_std': '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), max(standard_deviant_rates)),
                   'f1_dev': '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), min(standard_deviant_rates)),
                   'f2_onset': '{}_{:.0f}+{}'.format(prefix, max(center_frequencies), 'ONSET' ),
                   'f2_std': '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), max(standard_deviant_rates)),
                   'f2_dev': '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), min(standard_deviant_rates))}

    # Fold each group and organizes in dictionary
    folded_sounds = {sound_type: working_signal.extract_epoch(epoch_tag) for sound_type, epoch_tag in sound_types.items()}

    if return_clasified_responses == True:
        return folded_sounds
    elif return_clasified_responses == False:
        pass
    else:
        raise ValueError("return_clasified_ressonse sould be boolean")

    # calculates average response, i.e. PSTH
    avg_resp = {sound_type: np.nanmean(np.squeeze(folded_sound), axis=0)
                for sound_type, folded_sound in folded_sounds.items()}

    # selects the time window in which to calculate the SSA index, e.g. only during the sound presentation
    # this value is not explicint in the epochs, rather is necesary to take the difference between a whole
    # REFERENCE and the Pre and PostStimSilences

    eps = working_signal.epochs
    fs = working_signal.fs
    epoch_tags = ['REFERENCE', 'PreStimSilence', 'PostStimSilence']

    example_epoch_dict = dict.fromkeys(epoch_tags)
    for et in epoch_tags:
        example_epoch = eps.loc[eps['name']== et ,['start', 'end']].iloc[0,:]
        epoc_sample_len = int(np.round((example_epoch.end - example_epoch.start) * fs, decimals=0))
        example_epoch_dict[et] = epoc_sample_len

    SI_window = [example_epoch_dict['PreStimSilence'], # end of the PreStimSilence
                 example_epoch_dict['REFERENCE'] - example_epoch_dict['PostStimSilence']] # start of the PostStimSilence

    windowed_resp = {sound_type: psth[SI_window[0]: SI_window[1]] for sound_type, psth in avg_resp.items()}

    # integrates values across time
    integrated_resp = {sound_type: np.sum(win_resp) for sound_type, win_resp in windowed_resp.items()}

    # calculates different version of SSA index (SI)
    SSA_index_dict = dict.fromkeys(['f1', 'f2', 'cell'])

    for key, val in SSA_index_dict.items():
        # single frequencies SI
        if key in ['f1', 'f2']:
            std_key = '{}_{}'.format(key, 'std')
            dev_key = '{}_{}'.format(key, 'dev')
            SSA_index_dict[key] = ((integrated_resp[dev_key] - integrated_resp[std_key]) /
                                  (integrated_resp[dev_key] + integrated_resp[std_key]))
        # Full cell SI
        elif key == 'cell':
            SSA_index_dict[key] = ((integrated_resp['f1_dev'] + integrated_resp['f2_dev'] -
                                    integrated_resp['f1_std'] - integrated_resp['f2_std']) /
                                   (integrated_resp['f1_dev'] + integrated_resp['f2_dev'] +
                                    integrated_resp['f1_std'] + integrated_resp['f2_std']))

    return SSA_index_dict

def long_stim_to_point_stim(recording, scaling='same'):

    '''
        given a recordign, usually from an oddball experiment, takes the stimulus envelope "stim" and transforms it into
        a rasterized point process. This is a rasterized signal with the same dimetions as the original signal, but the
        stimulus appear as single ones in time marking their onset.

        Parameters
        ----------
        recording : A Recording object
            Generally the output of a model loader??

        scaling : 'same' or positive number
            'same' sets the amplitude of the onset valued equal to me maximum amplitud of the original stimulus
             or set scaling value to any arbitrary positive number

        Returns
        -------
        recording : A Recording object
            The same recording as the input, with the 'stim' signal modified as a rasterized point process

        '''

    stim = recording['stim']

    if isinstance(stim, signal.TiledSignal):
        stim = stim.rasterize()
    elif isinstance(stim, signal.RasterizedSignal):
        pass
    else:
        raise ValueError("recording['stim'] is not a tiled or rasterized signal")

    stim_as_matrix = stim.as_continuous()

    # Gets maximun value for later normalization
    original_max = np.nanmax(stim_as_matrix)

    if stim_as_matrix.shape[0] != 2:
        raise NotImplementedError ("more than two stimulation channels not yet supported")

    # makes NANs into 0 to allowe diff working
    nonan_matrix = copy.deepcopy(stim_as_matrix)
    nonan_matrix[np.isnan(stim_as_matrix)] = 0

    # gets the difference, padds with once 0 at the beginnign ot keep shape
    diff_matrix = np.diff(nonan_matrix, axis=1)
    diff_matrix = np.insert(diff_matrix, 0, 0, axis = 1)

    # foces all positive values to one, the envelope might not be a square envelope. This generalizes all envelopes
    # into a square form
    square_tone_matrix = diff_matrix > 0

    # find tone onsets
    onset_matrix = np.diff(square_tone_matrix.astype(float), axis=1)
    onset_matrix = np.insert(onset_matrix, 0, 0, axis=1)
    onset_matrix = onset_matrix == 1
    onset_matrix = onset_matrix.astype(float)

    # scales to the desired amplitud
    if scaling == 'same':
        onset_matrix = onset_matrix * original_max
    elif scaling > 0:
        onset_matrix = onset_matrix * scaling
    else:
        raise ValueError("scaling should be either 'same' or a positive number")

    # recover that nan values
    nan_mask = np.isnan(stim_as_matrix)
    onset_matrix[nan_mask] = np.nan

    point_stim = stim._modified_copy(onset_matrix)
    point_stim.name = 'stim'
    recording.add_signal(point_stim)
    rec = recording

def response_level (signal, metric='z_score'):
    #TODO finish documentation
    '''
    given a recording object, usually from an oddball experiment, determines the responsiveness of the celle to
    the different sound types in the oddball context,

    Parameters
    ----------
    signal : A signal object

    metric : string ['z_score']
        name of the metric calculated

    Returns
    -------
    SSA_index_dict : dict
        Dictionary containing the response level values for each of the sound frequency channels
    '''
    # TODO implement

    oddball_signal = set_oddball_epochs(signal)
    signal_mean = np.nanmean(oddball_signal.as_continuous())
    signal_std = np.nanstd(oddball_signal.as_continuous())

    #folds the data into
    sub_epochs = dict.fromkeys(['Sound', 'PreStimSilence', 'PostStimSilence'])
    for key in sub_epochs.keys():
        sub_epochs[key] = fold_oddball_signal(oddball_signal, sub_epoch=key)

    # get the mean of silent periods



    folded_sound = fold_oddball_signal(oddball_signal, sub_epoch='Sound')
    folded_prestim = fold_oddball_signal(oddball_signal, sub_epoch='Sound')
    folded_poststim


    folded_responses  = SSA_index(signal, subset='response', return_clasified_responses=True)


    rec = signal
    met = metric

    act_lvl_dict = {}

    return act_lvl_dict

def fast_plot(matrix):
    fig, ax = plt.subplots()
    ax.plot(matrix[0,:], marker='o')




# for thesting purposes, imports a pickled context containing recording objects and modelspecs
####### SSA_index #######
test_ctx = jl.load('/home/mateo/NeuPopFit/pickles/180531_test_context_full')
recording = test_ctx['val'][0]
subset = 'resp'
output = SSA_index(recording, subset='resp')

####### compare folding approaches #######
fold1 = SSA_index(recording, subset='resp', return_clasified_responses=True)
resp = recording['resp']
fold2 = fold_oddball_signal(set_oddball_epochs(resp))

for key in fold1.keys():
    x = fold1[key].flatten()
    y = fold2[key].flatten()
    fig, ax = plt.subplots()
    ax.scatter(x, y)
# its working

####### long_stim_to_point_stim #######
loaded_ctx = jl.load('/home/mateo/NeuPopFit/pickles/180601_test_context_only_load')
fig, ax = plt.subplots()
val = loaded_ctx['val']
stim = val.get_signal('stim')
ax.plot(stim.as_continuous()[0,:])
long_stim_to_point_stim(val)
stim = val.get_signal('stim')
ax.plot(stim.as_continuous()[0,:])

####### response_level #######



