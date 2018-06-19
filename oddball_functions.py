
import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal
import copy

# Functions working on signal objects

def set_signal_oddball_epochs(signal):
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
    prefix = epoch_names_to_extract[0].split('_')[0]  # STIM or PreStim

    # Checks the input signal has the adecuates frequencies and rates dimentions
    if (len(standard_deviant_rates) != 2) or (len(center_frequencies) != 2):
        raise ValueError("epochs contain {} presetnations rates, 2 are required\n"
                         "epochs contain {} center frequencies, 2 are required".format(len(standard_deviant_rates),
                                                                                       len(center_frequencies)))

    # explicit statement of dict key to event tag mapping
    key_mapping = {'{}_{:.0f}+{}'.format(prefix, min(center_frequencies), 'ONSET'): 'f1_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), max(standard_deviant_rates)): 'f1_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), min(standard_deviant_rates)): 'f1_dev',
                   '{}_{:.0f}+{}'.format(prefix, max(center_frequencies), 'ONSET'): 'f2_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), max(standard_deviant_rates)): 'f2_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), min(standard_deviant_rates)): 'f2_dev'}

    # creates new event dataframe with modified epoch names
    oddball_epochs = copy.deepcopy(signal.epochs)
    for oldkey, newkey in key_mapping.items():
        oddball_epochs.name.replace(oldkey, newkey, inplace=True)

    # Extract relevant subepochs as matrixe to check for intersections.
    sub_epochs_keys = ['PreStimSilence', 'PostStimSilence']
    sub_epochs_dict = {key: oddball_epochs.loc[
        oddball_epochs.name == key, ['start', 'end']
    ].sort_values(['end']).as_matrix()
                       for key in sub_epochs_keys}

    # checks size congruence of  PreStimSilence and PostStimSilence
    if sub_epochs_dict['PreStimSilence'].shape != sub_epochs_dict['PostStimSilence'].shape:
        raise ValueError('there are not equal number of PreStimSilence and PostStimSielence epochs')

    # chechs non overlaping of PreStimSilence and PostStimSilence
    silence_interesection = ep.epoch_intersection(sub_epochs_dict['PreStimSilence'], sub_epochs_dict['PostStimSilence'])
    if silence_interesection.size == 0:
        raise ValueError("there is overlap between PreStimSilence and PostStimSilence")

    # uses PreStimSilence ends and PostStimSilence starts to define Sound epochs
    sub_epochs_dict['Stim'] = np.stack([sub_epochs_dict['PreStimSilence'][:, 1],
                                        sub_epochs_dict['PostStimSilence'][:, 0]],
                                       axis=1)
    # add Stim as is to the data frame
    Stim_df = pd.DataFrame(sub_epochs_dict['Stim'], columns=['start', 'end'])
    Stim_df['name'] = 'Stim'
    oddball_epochs = oddball_epochs.append(Stim_df, ignore_index=True)

    # iterates over every oddball epoch and saves the intersectiosn with PreStimSilence Stim and PostStimSilence
    for _, oddball_key in key_mapping.items():
        oddball_matrix = oddball_epochs.loc[oddball_epochs.name == oddball_key, ['start', 'end']
        ].sort_values(['end']).as_matrix()

        for sub_epoch_name, sub_epoch_matrix in sub_epochs_dict.items():
            oddball_subepoch_matrix = ep.epoch_intersection(sub_epoch_matrix, oddball_matrix)
            oddball_subepoch_name = '{}_{}'.format(oddball_key, sub_epoch_name)

            # concatenates new oddball_subepochs into the oddball epochs DB
            df = pd.DataFrame(oddball_subepoch_matrix, columns=['start', 'end'])
            df['name'] = oddball_subepoch_name
            oddball_epochs = oddball_epochs.append(df, ignore_index=True)

    # Sort epochs by time and resets indexes
    oddball_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    oddball_epochs.reset_index(drop=True, inplace=True)

    # updates epochs in form original signal into new signal
    updated_signal = signal._modified_copy(signal._data, epochs=oddball_epochs)

    # something is not working with the epoch_intersection method
    return updated_signal


def extract_signal_oddball_epochs(signal, sub_epoch):
    '''
    returns a dictionary of the data matching each element in the usual Oddball epochs.

    Parameters
    ----------
    signal : A signal object
    sub_epoch : None, str, list of str
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

    # extract either one subepochs or multiple subepochs and concatenates\
    if isinstance(sub_epoch, list):
        folded_signal = dict.fromkeys(oddball_epoch_names)
        for this_odd_name in oddball_epoch_names:
            # extract each oddball specific subepoch and  hold in list
            subepoch_matrixes = list()
            for this_subep_name in sub_epoch:
                composite_name = '{}_{}'.format(this_odd_name, this_subep_name)
                print(composite_name)
                this_subep_mat = oddball_signal.extract_epoch(composite_name)
                print('shape: {}'.format(this_subep_mat.shape))
                subepoch_matrixes.append(this_subep_mat)
            # concatenate teh subepochs matrixes across time, i.e. third dimention
            folded_signal[this_odd_name] = np.concatenate(subepoch_matrixes, axis=2)

    else:
        # selects wich subepoch to extract, if any.
        if sub_epoch in sub_epochs_names:
            oddball_epoch_names = ['{}_{}'.format(this_ep_name, sub_epoch) for this_ep_name in oddball_epoch_names]
        elif sub_epoch == None:
            pass
        else:
            raise ValueError("sub_epoch has to be None, 'PreStimSilence', 'Stim' or 'PostStimSilence'")

        # Extract the folded epochs in an orderly dictionary.
        folded_signal = oddball_signal.extract_epochs(oddball_epoch_names)
        # Renames the dictionary key to the original oddball_epoch names
        folded_signal = {key.rsplit('_', 1)[0]: val for key, val in folded_signal.items()}

    return folded_signal


def get_signal_SI(signal, sub_epoch):
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

    folded_oddball = extract_signal_oddball_epochs(signal, sub_epoch=sub_epoch)

    # calculates PSTH
    PSTHs = {oddball_epoch_name: np.nanmean(np.squeeze(epoch_data), axis=0)
             for oddball_epoch_name, epoch_data in folded_oddball.items()}

    # integrates values across time
    integrated_resp = {sound_type: np.sum(win_resp) for sound_type, win_resp in PSTHs.items()}

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


def get_signal_activity(signal, sub_epoch, baseline='silence', metric='z_score'):
    '''
    given a signal object, usually from an oddball experiment, determines the responsiveness of the neuron to
    the different sound types in the oddball context,

    Parameters
    ----------
    signal : A signal object

    sub_epoch : None, str
        if none, considers the whole REFERENCE epoch, otherwise uses only the "sub epoch" contained within REFERENCE
        it should be 'Stim', 'PreStimSilence' or 'PostStimSilence'
    metric : string ['z_score']
        name of the metric calculated
        right now only one default option
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
    # frequencies (f1, f2)...

    # extract epochs names matching
    regexp = r"((SubStim)|(STIM))_\d*\+((0\.\d*)|ONSET)"
    epoch_names_to_extract = ep.epoch_names_matching(signal.epochs, regexp)

    # concatenates all sound types (onse, std, dev) by their frequency
    folded_oddball = extract_signal_oddball_epochs(signal, sub_epoch=sub_epoch)
    pooled_by_freq = dict.fromkeys(['f1', 'f2'])
    for key in pooled_by_freq.keys():
        pooled_by_freq[key] = np.concatenate([sound_data for sound_type, sound_data in folded_oddball.items() if
                                              sound_type.startswith(key)])

    # get the PSTH for each of the frequencies
    avg_resp = {frequency: np.nanmean(np.squeeze(folded_sound), axis=0)
                for frequency, folded_sound in pooled_by_freq.items()}

    # calculates the z_score for each of the frequencies and time bins
    z_scores = {frequency: (psth - base_mean) / base_std for frequency, psth in avg_resp.items()}

    # checks for significant time bins and averages z-scores, i.e.  z < -1.645 or 1.645 < z
    mean_significant_bins = {frequency: np.nanmean(z[np.logical_or(z < -1.645, z > 1.645)])
                             for frequency, z in z_scores.items()}

    return mean_significant_bins


# functions working on recordings objects

def as_rasterized_point_process(recording, scaling='same'):
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
        raise NotImplementedError("more than two stimulation channels not yet supported")

    # makes NANs into 0 to allowe diff working
    nonan_matrix = copy.deepcopy(stim_as_matrix)
    nonan_matrix[np.isnan(stim_as_matrix)] = 0

    # gets the difference, padds with once 0 at the beginnign ot keep shape
    diff_matrix = np.diff(nonan_matrix, axis=1)
    diff_matrix = np.insert(diff_matrix, 0, 0, axis=1)

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
    return recording


def get_recording_SI(recording, sub_epoch):
    signals = recording.signals

    # for SI calculates only for resp and pred, checks if both signals are in the recording
    relevant_keys = ['resp', 'pred']
    if set(relevant_keys).issubset(signals.keys()):
        pass
    else:
        raise ValueError("The recording does not have 'resp' and 'pred' signals")

    SI_dict = {sig_key: get_signal_SI(signal, sub_epoch) for sig_key, signal in signals.items() if
               sig_key in relevant_keys}

    return SI_dict


def get_recording_activity(recording, sub_epoch, baseline='silence'):
    signals = recording.signals

    # for activity calculates only for resp and pred, checks if both signals are in the recording
    relevant_keys = ['resp', 'pred']
    if set(relevant_keys).issubset(signals.keys()):
        pass
    else:
        raise ValueError("The recording does not have 'resp' and 'pred' signals")

    resp_dict = {sig_key: get_signal_activity(signal, sub_epoch, baseline=baseline) for sig_key, signal in
                 signals.items() if sig_key in relevant_keys}

    return resp_dict

#### graveyard ####

def SSA_index(recording, subset='resp', return_clasified_responses=False):
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

    prefix = epoch_names_to_extract[0].split('_')[0]  # STIM or PreStim

    # explicit statement of dict key to event tag mapping
    sound_types = {'f1_onset': '{}_{:.0f}+{}'.format(prefix, min(center_frequencies), 'ONSET'),
                   'f1_std': '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), max(standard_deviant_rates)),
                   'f1_dev': '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), min(standard_deviant_rates)),
                   'f2_onset': '{}_{:.0f}+{}'.format(prefix, max(center_frequencies), 'ONSET'),
                   'f2_std': '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), max(standard_deviant_rates)),
                   'f2_dev': '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), min(standard_deviant_rates))}

    # Fold each group and organizes in dictionary
    folded_sounds = {sound_type: working_signal.extract_epoch(epoch_tag) for sound_type, epoch_tag in
                     sound_types.items()}

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
        example_epoch = eps.loc[eps['name'] == et, ['start', 'end']].iloc[0, :]
        epoc_sample_len = int(np.round((example_epoch.end - example_epoch.start) * fs, decimals=0))
        example_epoch_dict[et] = epoc_sample_len

    SI_window = [example_epoch_dict['PreStimSilence'],  # end of the PreStimSilence
                 example_epoch_dict['REFERENCE'] - example_epoch_dict[
                     'PostStimSilence']]  # start of the PostStimSilence

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


def set_signal_oddball_epochs_v2(signal):
    # this is slow as shit!

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
    prefix = epoch_names_to_extract[0].split('_')[0]  # STIM or PreStim

    # Checks the input signal has the adecuates frequencies and rates dimentions
    if (len(standard_deviant_rates) != 2) or (len(center_frequencies) != 2):
        raise ValueError("epochs contain {} presetnations rates, 2 are required\n"
                         "epochs contain {} center frequencies, 2 are required".format(len(standard_deviant_rates),
                                                                                       len(center_frequencies)))

    # explicit statement of dict key to event tag mapping
    key_mapping = {'{}_{:.0f}+{}'.format(prefix, min(center_frequencies), 'ONSET'): 'f1_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), max(standard_deviant_rates)): 'f1_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), min(standard_deviant_rates)): 'f1_dev',
                   '{}_{:.0f}+{}'.format(prefix, max(center_frequencies), 'ONSET'): 'f2_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), max(standard_deviant_rates)): 'f2_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), min(standard_deviant_rates)): 'f2_dev'}

    # creates new event dataframe with modified epoch names
    oddball_epochs = copy.deepcopy(signal.epochs)
    for oldkey, newkey in key_mapping.items():
        oddball_epochs.name.replace(oldkey, newkey, inplace=True)

    # Using the epochs DF, interates over every oddball_ epoch.
    PreStim_mask = oddball_epochs.name == 'PreStimSilence'
    PostStim_mask = oddball_epochs.name == 'PostStimSilence'

    for _, oddball_key in key_mapping.items():
        epoch_mask = oddball_epochs.name == oddball_key
        current_epochs = oddball_epochs.loc[epoch_mask, :]

        for ii, this_oddball in current_epochs.iterrows():
            # get start time and end time and checks what PreStimSilence and PostStimSilence are contained
            start_mask = oddball_epochs.start >= this_oddball.start
            end_mask = oddball_epochs.end <= this_oddball.end
            contained_epochs = oddball_epochs.loc[start_mask & end_mask, :]
            # extract contained Pre and Post StimSilence
            PreStim_epoch = oddball_epochs.loc[start_mask & end_mask & PreStim_mask, :].iloc[0, :]
            PostStim_epoch = oddball_epochs.loc[start_mask & end_mask & PostStim_mask, :].iloc[0, :]
            # Modify name and append copy
            for silence in [PreStim_epoch, PostStim_epoch]:
                compound_name = '{}_{}'.format(this_oddball['name'], silence['name'])
                renamed_subepoch = pd.Series([silence.start, silence.end, compound_name],
                                             ['start', 'end', 'name'])
                oddball_epochs = oddball_epochs.append(renamed_subepoch, ignore_index=True, verify_integrity=False)

            new_stim_name = '{}_Stim'.format(this_oddball['name'])
            new_Stim = pd.Series([PreStim_epoch.end, PostStim_epoch.start, new_stim_name],
                                 ['start', 'end', 'name'])
            oddball_epochs = oddball_epochs.append(new_Stim, ignore_index=True, verify_integrity=False)

    oddball_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    oddball_epochs.reset_index(drop=True, inplace=True)

    # updates epochs in form original signal into new signal
    updated_signal = signal._modified_copy(signal._data, epochs=oddball_epochs)

    # something is not working with the epoch_intersection method
    return updated_signal


def set_signal_oddball_epochs_v3(signal):
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
    prefix = epoch_names_to_extract[0].split('_')[0]  # STIM or PreStim

    # Checks the input signal has the adecuates frequencies and rates dimentions
    if (len(standard_deviant_rates) != 2) or (len(center_frequencies) != 2):
        raise ValueError("epochs contain {} presetnations rates, 2 are required\n"
                         "epochs contain {} center frequencies, 2 are required".format(len(standard_deviant_rates),
                                                                                       len(center_frequencies)))

    # explicit statement of dict key to event tag mapping
    key_mapping = {'{}_{:.0f}+{}'.format(prefix, min(center_frequencies), 'ONSET'): 'f1_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), max(standard_deviant_rates)): 'f1_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, min(center_frequencies), min(standard_deviant_rates)): 'f1_dev',
                   '{}_{:.0f}+{}'.format(prefix, max(center_frequencies), 'ONSET'): 'f2_onset',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), max(standard_deviant_rates)): 'f2_std',
                   '{}_{:.0f}+{:.2f}'.format(prefix, max(center_frequencies), min(standard_deviant_rates)): 'f2_dev'}

    # creates new event dataframe with modified epoch names
    oddball_epochs = copy.deepcopy(signal.epochs)
    for oldkey, newkey in key_mapping.items():
        oddball_epochs.name.replace(oldkey, newkey, inplace=True)

    # Using the epochs DF, interates over every oddball_ epoch.
    PreStim_mask = oddball_epochs.name == 'PreStimSilence'
    PostStim_mask = oddball_epochs.name == 'PostStimSilence'

    for _, oddball_key in key_mapping.items():
        epoch_mask = oddball_epochs.name == oddball_key
        current_epochs = oddball_epochs.loc[epoch_mask, :]

        for ii, this_oddball in current_epochs.iterrows():
            # get start time and end time and checks what PreStimSilence and PostStimSilence are contained
            start_mask = oddball_epochs.start >= this_oddball.start
            end_mask = oddball_epochs.end <= this_oddball.end
            contained_epochs = oddball_epochs.loc[start_mask & end_mask, :]
            # extract contained Pre and Post StimSilence
            PreStim_epoch = oddball_epochs.loc[start_mask & end_mask & PreStim_mask, :].iloc[0, :]
            PostStim_epoch = oddball_epochs.loc[start_mask & end_mask & PostStim_mask, :].iloc[0, :]
            # Modify name and append copy
            for silence in [PreStim_epoch, PostStim_epoch]:
                compound_name = '{}_{}'.format(this_oddball['name'], silence['name'])
                renamed_subepoch = pd.Series([silence.start, silence.end, compound_name],
                                             ['start', 'end', 'name'])
                oddball_epochs = oddball_epochs.append(renamed_subepoch, ignore_index=True, verify_integrity=False)

            new_stim_name = '{}_Stim'.format(this_oddball['name'])
            new_Stim = pd.Series([PreStim_epoch.end, PostStim_epoch.start, new_stim_name],
                                 ['start', 'end', 'name'])
            oddball_epochs = oddball_epochs.append(new_Stim, ignore_index=True, verify_integrity=False)

    oddball_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    oddball_epochs.reset_index(drop=True, inplace=True)

    # updates epochs in form original signal into new signal
    updated_signal = signal._modified_copy(signal._data, epochs=oddball_epochs)

    # something is not working with the epoch_intersection method
    return updated_signal


def epoch_intersection(a, b):
    '''
    Compute the intersection of the epochs. Only regions in a which overlap with
    b will be kept.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    intersection : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the difference of a and b.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result:  [  ]       [ ]             []
    '''
    # Convert to a list and then sort in reversed order such that pop() walks
    # through the occurences from earliest in time to latest in time.
    a = a.tolist()
    a.sort(reverse=True)
    b = b.tolist()
    b.sort(reverse=True)

    intersection = []
    if len(a) == 0 or len(b) == 0:
        # lists are empty, just exit
        result = np.array([])
        return result

    lb, ub = a.pop()
    lb_b, ub_b = b.pop()

    while True:
        if lb > ub_b:
            #           [ a ]
            #     [ b ]
            # Current epoch in b ends before current epoch in a. Move onto
            # the next epoch in b.
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif ub <= lb_b:
            #   [  a    ]
            #               [ b        ]
            # Current epoch in a ends before current epoch in b. Add bounds
            # and move onto next epoch in a.
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb == lb_b) and (ub == ub_b):
            #   [  a    ]
            #   [  b    ]
            # Current epoch in a matches epoch in b.
            try:
                intersection.append((lb, ub))
                lb, ub = a.pop()
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= ub_b):
            #   [  a    ]
            #     [ b ]
            # Current epoch in b is fully contained in the  current epoch
            # from a. Save everything in
            # a up to the beginning of the current epoch of b. However, keep
            # the portion of the current epoch in a
            # that follows the end of the current epoch in b so we can
            # detremine whether there are additional epochs in b that need
            # to be cut out..
            intersection.append((lb_b, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= lb_b) and (ub <= ub_b):
            #   [  a    ]
            #     [ b        ]
            # Current epoch in b begins in a, but extends past a.
            intersection.append((lb_b, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (lb <= ub_b):
            #   [  a    ]
            # [       b     ]
            # Current epoch in a is fully contained in b
            intersection.append((lb, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (ub < ub_b) and (lb > ub_b):
            #   [  a    ]
            # [ b    ]
            intersection.append((lb, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        else:
            # This should never happen.
            m = 'Unhandled epoch boundary condition. Contact the developers.'
            raise SystemError(m)

    # Add all remaining epochs from a
    # intersection.extend(a[::-1])
    result = np.array(intersection)
    if result.size == 0:
        raise Warning("Epochs did not intersect, resulting array"
                      " is empty.")
    return result


