import matplotlib.pyplot as plt
import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal
import copy
import nems_db.baphy as nb
import nems_db.db as db
import warnings


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

    # checks if the signal already has oddball epochs
    oddball_epochs_names = {'f1_onset', 'f1_std', 'f1_dev', 'f2_onset', 'f2_std', 'f2_dev'}
    signal_epochs_names = set(signal.epochs.name.unique())
    if oddball_epochs_names.issubset(signal_epochs_names):
        return signal
    else:
        pass

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
        raise ValueError("epochs contain {} presentation rates, 2 are required\n"
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
    ].sort_values(['end']).as_matrix() # Todo Solve the as_matrix replaced by value
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
        ].sort_values(['end']).as_matrix() # Todo Solve the as_matrix replaced by value

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

    # updates epochs in form original signal into a copy
    new_signal = signal._modified_copy(signal._data, epochs=oddball_epochs)

    return new_signal


def get_superepoch_subset(signal, super_epoch):
    '''
    returns a copy of the signal only with epochs contained entirely by super_epoch. i.e. an intersection of every epoch
    with super_epoch

    :param signal: A signal object from an oddball experiment, preferably after defining the oddbal subepochs with
                   set_signal_oddball_epochs()
    :param super_epoch: None, str or [str,...] with the name/names of the epochs
    :return: a epoch data frame
    '''
    # checks format of super_epoch
    if isinstance(super_epoch, str):
        super_epoch = [super_epoch]
    elif isinstance(super_epoch, list):
        pass
    elif super_epoch == None:
        # if empty list, asume no super_epochs desired and returns original signal epochs
        return signal.epochs
    else:
        raise ValueError('super_epochs must be a strig, list of strings or empty list')

    epochs = signal.epochs
    e_names = epochs.name.unique().tolist()

    for sup in super_epoch:
        # checks if superepochs are within the signal epochs
        if sup not in e_names:
            mesg = "super_epoch {} is not an epoch of the signal".format(sup)
            warnings.warn(RuntimeWarning(mesg))

    # get the equivalent np array of the specified 2D array of (M x 2), holds in dict to add later
    superepoch_dict = {sup: epochs.loc[epochs.name == sup, ['start', 'end']].values
                       for sup in super_epoch}

    # as a list for easier epoch union
    superepoch_list = [val for key, val in superepoch_dict.items()]

    # iteratively takes the union of all the superepochs
    while len(superepoch_list) >= 2:
        try:
            a = superepoch_list.pop()
            b = superepoch_list.pop()
            c = ep.epoch_union(a, b)
            superepoch_list.append(c)
        except IndexError:
            raise SystemError('this message should never show')

    super_epoch_union = superepoch_list[0]  # this should be a single item list

    # iterates over every other epoch that is not a super_epoch
    # and gets the intersection, storing in the dictionary under the same name
    e_names = dict.fromkeys([name for name in e_names if name not in sup])
    contained_epochs = {
    key: ep.epoch_intersection(epochs.loc[epochs.name == key, ['start', 'end']].values, super_epoch_union)
    for key, val in e_names.items()}

    # add the epochs corresponding to super_epoch
    cont_sup_epochs = {**contained_epochs, **superepoch_dict}

    # takes the arrays and key names and organize in an epoch dataframe
    new_epochs_list = list()

    for name, arr in cont_sup_epochs.items():
        if arr.shape == (0,):
            #ignores empty arrays TODO, check if this will break things downstream because of absent key
            # woudl not it be better to have an epoch with start = end = 0?
            continue
        cont_epoch = pd.DataFrame({
            'name': name,
            'start': arr[:, 0],
            'end': arr[:, 1]})

        new_epochs_list.append(cont_epoch)
    # concatenates all dataframes together
    new_epochs = pd.concat(new_epochs_list)

    # formats the dataframe: indexes, order by start time and a order columns for better readability
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)
    new_epochs = new_epochs[['start', 'end', 'name']]
    # TODO, figure out why longer epochs like PASSIVE_EXPERIMET are being broken down into shorter pieces when usign TRIAL as a superepoch.
    return new_epochs


def set_signal_jitter_epochs(signal):
    '''
    takes a signal from an ssa experiment, looks for epochs named after files, and renames those epochs
    as Jitter On or Jitter Off. if no file is found, asumes Jitter Off
    :param signal: a signal object from an oddball experiment
    :return: a copy of the signal object with epochs describing the Jitter status
    '''

    # regexp for experimetn name e.g. 'FILE_gus037d03_p_SSA' and 'FILE_gus037d04_p_SSA'
    regexp = r"\AFILE_\D{3}\d{3}\D\d{2}_p_SSA"
    epoch_names_to_extract = ep.epoch_names_matching(signal.epochs, regexp)
    if not epoch_names_to_extract:
        raise ValueError('no epochs coresponding to file names')
        # TODO handle when there is no matching epochs. i.e. only one jitter status.

    epoch_rename_map = dict.fromkeys(epoch_names_to_extract)
    cellid = signal.recording

    # related parmfiles
    parmfiles = db.get_batch_cell_data(cellid=cellid, batch=296)
    parmfiles = list(parmfiles['parm'])
    # creates a dictionary mapping the epoch keys to the parmfiles paths, i.e.
    # from: 'FILE_gus037d03_p_SSA' to: '/auto/data/daq/Augustus/gus037/gus037d03_p_SSA.m'
    parmfiles = {'FILE_{}'.format(path.split('/', )[-1]):
                     '{}.m'.format(path)
                 for path in parmfiles}

    jitter_status = list()
    # to relate filename to jitter status pulls experimetn parameters
    for oldkey, _ in epoch_rename_map.items():
        # todo this thing takes a lot of time. It should be implemented when generating the recording/signal epochs.
        globalparams, exptparams, exptevents = nb.baphy_parm_read(parmfiles[oldkey])

        # convoluted indexig into the nested dictionaries ot get Jitter status, sets value to "Off" by default
        j_stat = exptparams['TrialObject'][1]['ReferenceHandle'][1].get('Jitter', 'Off')
        epoch_rename_map[oldkey] = 'Jitter_{}'.format(j_stat.rstrip())

    # get epochs, rename as stated by the map
    new_epochs = signal.epochs.copy()
    for oldkey, newkey in epoch_rename_map.items():
        new_epochs.loc[new_epochs.name == oldkey, ['name']] = newkey

    new_signal = signal._modified_copy(signal._data, epochs=new_epochs)

    return new_signal


def extract_signal_oddball_epochs(signal, sub_epoch, super_epoch):
    '''
    returns a dictionary arrays  corresponding to stacked repetitions of each of the Oddball epochs.

    Parameters
    ----------
    signal : A signal object
    sub_epoch : None, str, list of str
        if none, returns the whole REFERENCE epoch, otherwise returns the "sub epoch" contained within REFERENCE
        it should be 'Stim', 'PreStimSilence' or 'PostStimSilence'
    super_epoch: None, str, list of str
        if none, returns  all REFERENCE possibel, otherwise returns REFERENCES(or sub epochs) contained within super_epoch
        in the context of the oddbal experiments it should be 'Jitter_ON', 'Jitter_Off' or 'Jitter_Both'

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

    # select subset of epochs contained in super_epoch
    super_epoch_subset = get_superepoch_subset(oddball_signal, super_epoch=super_epoch)
    # if super_epochs_subset is empty, returns a dict with the same structure as normal, but filled with empty arrays
    if super_epoch_subset.empty:
        mesg = 'no epochs contained in {}, returning dict with nan arrays '.format(str(super_epoch))
        warnings.warn(RuntimeWarning(mesg))
        folded_signal = {key: np.full((1,1), np.nan) for key in oddball_epoch_names}
        return folded_signal
    else:
        pass

    oddball_signal = oddball_signal._modified_copy(oddball_signal._data, epochs=super_epoch_subset)

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


def get_signal_SI(signal, sub_epoch, super_epoch):
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

    folded_oddball = extract_signal_oddball_epochs(signal, sub_epoch=sub_epoch, super_epoch=super_epoch)

    # Todo if folded_oddball is NaN, sets all values of SI to NaN

    # calculates PSTH
    PSTHs = {oddball_epoch_name: np.squeeze(np.nanmean(epoch_data, axis=0))
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


def get_signal_activity(signal, sub_epoch, super_epoch, baseline='silence', metric='z_score'):
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
    folded_oddball = extract_signal_oddball_epochs(signal, sub_epoch=sub_epoch, super_epoch=super_epoch)

    # Todo if folded_oddball is NaN, sets all values of SI to NaN


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


def signal_nan_as_zero(signal):
    '''
    takes any signal object (RasterizedSignal, TiledSignal) and returns an equivalent signal
    where NAN has been replaced with zeros.
    :param signal: a signal object
    :return: a copy of the input signal object sans NaNs
    '''
    arr = signal.rasterize().as_continuous().copy()
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0
    no_nan_signal = signal.rasterize()._modified_copy(arr)
    return no_nan_signal


# functions working on recordings objects

def as_rasterized_point_process(recording, scaling):
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

    # makes NANs into 0 to allow diff working
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


def get_recording_SI(recording, sub_epoch, super_epoch):
    signals = recording.signals

    # for SI calculates only for resp and pred, checks if both signals are in the recording
    relevant_keys = ['resp', 'pred']
    if set(relevant_keys).issubset(signals.keys()):
        pass
    else:
        raise ValueError("The recording does not have 'resp' and 'pred' signals")

    SI_dict = {sig_key: get_signal_SI(signal, sub_epoch, super_epoch) for sig_key, signal in signals.items() if
               sig_key in relevant_keys}

    return SI_dict


def get_recording_activity(recording, sub_epoch, super_epoch, baseline='silence'):
    signals = recording.signals

    # for activity calculates only for resp and pred, checks if both signals are in the recording
    relevant_keys = ['resp', 'pred']
    if set(relevant_keys).issubset(signals.keys()):
        pass
    else:
        raise ValueError("The recording does not have 'resp' and 'pred' signals")

    resp_dict = {sig_key: get_signal_activity(signal, sub_epoch, super_epoch, baseline=baseline) for sig_key, signal in
                 signals.items() if sig_key in relevant_keys}

    return resp_dict


def recording_nan_as_zero(recording, signals_subset=['stim']):
    '''
    for a given recordign, takes the signals of that recordign specified by signals_subset, and sets the nan values to
    zero, if signal subset is an empty list, performs the operation for all the signals in the recording.
    :param recording: a recording object
    :param signals_subset: an empty list of list of strings corresponding to the names of signals in the recording
    :return: a copy of the recording with modified signals
    '''
    if isinstance(signals_subset, str):
        signals_subset = [signals_subset]
    elif signals_subset == None:
        signals_subset = list(recording.signals.keys())
    elif isinstance(signals_subset, list):
        pass
    else:
        return ValueError("'signals_subset must be str, [str,...] or None")

    new_recording = recording.copy()

    for select_sig in signals_subset:
        if not isinstance(select_sig, str):
            raise ValueError("signal_subset must be a list of strings")
        elif select_sig not in recording.signals.keys():
            raise ValueError("{} is not a signal of the recording".format(select_sig))
        else:
            pass

        signal = new_recording[select_sig]
        new_signal = signal_nan_as_zero(signal)
        new_signal.name = select_sig
        new_recording.add_signal(new_signal)

    return new_recording


def set_recording_jitter_epochs(recording):
    new_recording = recording.copy()
    for name, signal in recording.signals.items():
        new_signal = set_signal_jitter_epochs(signal)
        new_recording[name] = new_signal
    return new_recording


def set_recording_oddball_epochs(recording):
    new_recording = recording.copy()
    for name, signal in recording.signals.items():
        new_signal = set_signal_oddball_epochs(signal)
        new_recording[name] = new_signal
    return new_recording


# data base interfacing functions

def get_oddball_parmfiles(cellid):
    parmfiles = db.get_batch_cell_data(batch=296, cellid=cellid, rawid=None, label=None)