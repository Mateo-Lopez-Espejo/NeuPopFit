import joblib as jl
import nems.epoch as ep
import copy
import numpy as np
import matplotlib.pyplot as plt

def set_signal_oddball_epochs(signal):
    # TODO should this be implemented for recordigns instead of signals, What is the difference between signal and recordiing epochs?
    '''
    rename the signal epochs to a form generalized for oddball stimulus.
    i.e. (onset, standard, deviant) * (frequency 1, frequency 2).
    Adds a 'Stim' epoch in between the 'PreStimSilence' and 'PostStimSilence'

    Parameters
    ----------
    signal: a signal object from an oddball experimetn

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

test_file_path = '/auto/users/mateo/180606_oddball_ctx'

ctx = jl.load(test_file_path)
sig = ctx['val'][0]['resp']


odd_sig = set_signal_oddball_epochs(sig)
epochs = odd_sig.epochs
# epochs should have not collitions, every 'REFERENCE' (Which exactly correspond to the oddball epochs: f1_std, f2_dev ...)
# should contain a 'PreStimSilence', 'Stim' and 'PostStimSilence'

epochs.head(20)

# but if I try to extrac the 'Stim' of every f1_std, the resulting slices are longer than 'Stim' and even

# the Whole REFERENCE of an specific oddball tag
f1_std = odd_sig.extract_epoch('f1_std')

# the Stim independent from the oddball tag
Stim = odd_sig.extract_epoch('Stim')

# the intersection of the two previous epochs. This should give the subset of 'Stim' that are specific for the Oddball tag
f1_std_stim = odd_sig.extract_epoch('Stim', overlapping_epoch='f1_std')

# the time dimention is inconsitent:
print('f1_std: {}\nall_stim: {}\nf1_std_stim: {}'.format(f1_std.shape, Stim.shape, f1_std_stim.shape))

fig, ax = plt.subplots()

toplot= {'f1_std':f1_std, 'Stim':Stim, 'f1_std_stim':f1_std_stim}

for key, obj in toplot.items():
    psth = np.nanmean(obj,axis=0).squeeze()
    ax.plot(psth, label=key)
ax.legend()
