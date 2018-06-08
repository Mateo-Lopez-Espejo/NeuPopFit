
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


def set_signal_oddball_epochs(signal):
    # TODO this should be implemented for recordigns instead of signals, What is the difference between signal and recordiing epochs?
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

    # updates epochs in form original signal into new signal
    updated_signal = signal._modified_copy(signal._data, epochs=oddball_epochs)

    # generates new events containing the sound time
    new_event_name = 'Stim'

    # checks size congruence of  PreStimSilence and PostStimSilence
    PreStimEnd = oddball_epochs.loc[oddball_epochs.name == 'PreStimSilence', ['end']].sort_values(['end'])
    PostStimStart = oddball_epochs.loc[oddball_epochs.name == 'PostStimSilence', ['start']].sort_values(['start'])
    if PreStimEnd.shape == PostStimStart.shape:
        pass
    else:
        raise ValueError('there are not equal number of PreStimSilence and PostStimSielence epochs')

    # chechs non overlaping of PreStimSilence and PostStimSilence
    time_difference = PostStimStart.as_matrix() - PreStimEnd.as_matrix()
    if np.any(time_difference<=0):
        raise ValueError("there is overlap between PreStimSilence and PostStimSilence")

    # uses PreStimSilence ends and PostStimSilence starts to define Sound epochs
    sound_epoch_matrix = np.stack([PreStimEnd.as_matrix().squeeze(), PostStimStart.as_matrix().squeeze()], axis=1)
    updated_signal.add_epoch(new_event_name, sound_epoch_matrix)

    #sort epochs Todo ask brad why is this sorting not already implemented in signal.add_epoch
    updated_signal.epochs.sort_values(by=['start', 'end'], ascending=[True,False], inplace=True)
    updated_signal.epochs.reset_index(drop=True, inplace=True)

    return updated_signal

def get_sound_window_index(signal):
    '''
    Returns the indexes for the folded oddball signal np.arrays, corresponding to the begining and end of the epoch
    :param signal: an oddball signal object
    :return: list of two indexes
    '''
    eps = signal.epochs
    fs = signal.fs
    epoch_tags = ['REFERENCE', 'PreStimSilence', 'PostStimSilence']

    example_epoch_dict = dict.fromkeys(epoch_tags)
    for et in epoch_tags:
        example_epoch = eps.loc[eps['name']== et ,['start', 'end']].iloc[0,:]
        epoc_sample_len = int(np.round((example_epoch.end - example_epoch.start) * fs, decimals=0))
        example_epoch_dict[et] = epoc_sample_len

    window_indexes = [example_epoch_dict['PreStimSilence'], # end of the PreStimSilence
                 example_epoch_dict['REFERENCE'] - example_epoch_dict['PostStimSilence']] # start of the PostStimSilence
    return window_indexes

def extract_signal_oddball_epochs(signal, sub_epoch = None):
    '''
    returns a dictionary of the data matching each element in the usual Oddball epochs.

    Parameters
    ----------
    signal : A signal object
    sub_epoch : None, str
        if none, returns the whole REFERENCE epoch, otherwise returns the "sub epoch" contained within REFERENCE
        it should be 'Stim', 'PreStimSilence' or 'PostStimSilence'

    Returns
    -------
    folded_signal : dict
        Dictionary containing (M x C x N) matrixes with values for each sound type, where M is the repetition,
        C is the channel adn N is time

    '''

    signal = signal.rasterize()
    oddball_signal = set_signal_oddball_epochs(signal)
    oddball_epoch_names = ['f1_onset', 'f1_std', 'f1_dev', 'f2_onset', 'f2_std', 'f2_dev']
    sub_epochs_names = ['PreStimSilence', 'PostStimSilence', 'Stim']

    if sub_epoch == None:
        # returns the the full reference
        folded_signal = oddball_signal.extract_epochs(oddball_epoch_names)
    elif sub_epoch in sub_epochs_names:
        # todo this righr now is not working for unknonw reasons. Something with the epochs or the function, ask stephen and brad respectiely
        folded_signal = {epoch_name: oddball_signal.extract_epoch(sub_epoch, overlapping_epoch=epoch_name)
                         for epoch_name in oddball_epoch_names}
    else:
        raise ValueError("sub_epoch shoudl be None 'PreStimSilence' 'Stim' or 'PostStimSilence'")

    return folded_signal

def get_signal_SI(signal):

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

    get_sound_window_index(signal)
    folded_oddball = extract_signal_oddball_epochs(signal)

    # calculates PSTH
    PSTHs = {oddball_epoch_name: np.nanmean(np.squeeze(epoch_data), axis=0)
                for oddball_epoch_name, epoch_data in folded_oddball.items()}


    SI_window = get_sound_window_index(signal)
    widowed_PSTHs = {sound_type: psth[SI_window[0]: SI_window[1]] for sound_type, psth in PSTHs.items()}

    # integrates values across time
    integrated_resp = {sound_type: np.sum(win_resp) for sound_type, win_resp in widowed_PSTHs.items()}

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

    SI_window = get_sound_window_index(working_signal)


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

def response_level (signal, metric='z_score', baseline='silence'):
    #TODO finish documentation
    '''
    given a recording object, usually from an oddball experiment, determines the responsiveness of the celle to
    the different sound types in the oddball context,

    Parameters
    ----------
    signal : A signal object

    metric : string ['z_score']
        name of the metric calculated

    baseine : str
        'signal' uses the std and mean form the whole signal for z-score calculation
        'silence' uses the std and mean form the PreStimSilence for z-score calculation

    Returns
    -------
    SSA_index_dict : dict
        Dictionary containing the response level values for each of the sound frequency channels
    '''

    if baseline == 'signal':
        base_mean = np.nanmean(signal.as_continuous())
        base_std = np.nanstd(signal.as_continuous())
    elif baseline == 'silence':
        PreStimSilence = signal.extract_epoch('PreStimSilence')
        base_mean = np.nanmean(PreStimSilence)
        base_std = np.nanstd(PreStimSilence)
    else:
        raise ValueError("unsuported baseline parameter. Use 'signal' or 'silence' ")

    # the "z_score" is calculated for each time bin from the average of the 'Sound' epochs for each of the stimulation
    # frequencies (f1, f2)

    # concatenates all sound types (onse, std, dev) by their frequency
    folded_oddball = extract_signal_oddball_epochs(signal, sub_epoch=None)
    pooled_by_freq = dict.fromkeys(['f1', 'f2'])
    for key in pooled_by_freq.keys():
        pooled_by_freq[key] = np.concatenate([sound_data for sound_type, sound_data in folded_oddball.items() if
                                            sound_type.startswith(key)])

    # get the PSTH for each of the frequencies
    avg_resp = {frequency: np.nanmean(np.squeeze(folded_sound), axis=0)
                for frequency, folded_sound in pooled_by_freq.items()}

    # calculates the z_score for each of the frequencies and time bins
    z_scores = {frequency: (psth - base_mean)/base_std for frequency, psth in avg_resp.items()}


    # selects the only the sound window (excludes pre and poststim silences)
    soundwindow = get_sound_window_index(signal)
    windowed_score = {frequency: z[soundwindow[0]:soundwindow[1]] for frequency, z in z_scores.items()}

    # checks for significant time bins and averages z-scores, i.e.  z < -1.645 or 1.645 < z
    mean_significant_bins = {frequency: np.nanmean(z[np.logical_or(z < -1.645 , z > 1.645)])
                             for frequency, z in windowed_score.items()}

    return mean_significant_bins




# for thesting purposes, imports a pickled context containing recording objects and modelspecs
####### SSA_index #######
test_ctx = jl.load('/home/mateo/NeuPopFit/pickles/180531_test_context_full')
recording = test_ctx['val'][0]
subset = 'resp'
output = SSA_index(recording, subset='resp')

####### compare folding approaches #######
fold1 = SSA_index(recording, subset='resp', return_clasified_responses=True)
resp = recording['resp']
fold2 = extract_signal_oddball_epochs(set_signal_oddball_epochs(resp))

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

####### plot folded ######
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for key, obj in folded_sounds.items():
    psth = np.nanmean(obj, axis=0)
    ax.plot(psth.squeeze())

