import os
import logging
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import nems
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import itertools as itt
from nems.recording import Recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
import nems_db.db as db
import nems_db.baphy as nb
import pandas as pd
import nems_db.xform_wrappers as nw


# Import options, i am not sure if i should be importing anything else.
# Importing stim generates an error, probably because the stim is an envelope rather than a sprectrogram
# what rasterfs to use? now that i think about it, this raster freque might be the issue with individual tone
# tone presentations not appearing in the event list, although it should not be, given that the tones are 100ms (10 Hz)
# and the sampling rate is twice as much (20 hz).
options = {'rasterfs': 20,
           'stim': False,
           'includeprestim': True,
           'stimfmt': 'none',
           'chancount': 0}

cellid = 'gus037d-a2'
batch = 296

# loading the recording with this method is somehow overwriting the events.
rec = nb.baphy_load_recording(cellid, batch, options)
# the issue is in baphy_load_dataset call in line 1055 of nems_db/baphy
# here i am changing that line to point to my own verison of the function:
# baphy_load_dataset_mateo, defined later.

def baphy_load_recording_mateo(cellid, batch, options):
    # print(options)
    options['rasterfs'] = int(options.get('rasterfs', 100))
    options['stimfmt'] = options.get('stimfmt', 'ozgf')
    options['chancount'] = int(options.get('chancount', 18))
    options['pertrial'] = int(options.get('pertrial', False))
    options['includeprestim'] = options.get('includeprestim', 1)

    options['pupil'] = int(options.get('pupil', False))
    options['pupil_deblink'] = int(options.get('pupil_deblink', 1))
    options['pupil_median'] = int(options.get('pupil_deblink', 1))
    options['stim'] = int(options.get('stim', True))
    options['runclass'] = options.get('runclass', None)
    options['cellid'] = options.get('cellid', cellid)
    options['batch'] = int(batch)

    d = db.get_batch_cell_data(batch=batch, cellid=cellid, label='parm')
    if len(d)==0:
        raise ValueError('cellid/batch entry not found in NarfData')

    files = list(d['parm'])

    for i, parmfilepath in enumerate(files):

        if options["runclass"] == "RDT":
            event_times, spike_dict, stim_dict, \
                state_dict, stim1_dict, stim2_dict = \
                nb.baphy_load_dataset_RDT(parmfilepath, options)
        else:
            event_times, spike_dict, stim_dict, state_dict = \
                baphy_load_dataset_mateo(parmfilepath, options)

            d2 = event_times.loc[0].copy()
            if (i == 0) and (d2['name'] == 'PASSIVE_EXPERIMENT'):
                d2['name'] = 'PRE_PASSIVE'
                event_times = event_times.append(d2)
            elif d2['name'] == 'PASSIVE_EXPERIMENT':
                d2['name'] = 'POST_PASSIVE'
                event_times = event_times.append(d2)

        # generate spike raster
        raster_all, cellids = nb.spike_time_to_raster(
                spike_dict, fs=options['rasterfs'], event_times=event_times
                )

        # generate response signal
        t_resp = nems.signal.RasterizedSignal(
                fs=options['rasterfs'], data=raster_all, name='resp',
                recording=cellid, chans=cellids, epochs=event_times
                )
        if i == 0:
            resp = t_resp
        else:
            resp = resp.concatenate_time([resp, t_resp])

        if options['pupil']:

            # create pupil signal if it exists
            rlen = int(t_resp.ntimes)
            pcount = state_dict['pupiltrace'].shape[0]
            plen = state_dict['pupiltrace'].shape[1]
            if plen > rlen:
                state_dict['pupiltrace'] = \
                    state_dict['pupiltrace'][:, 0:-(plen-rlen)]
            elif rlen > plen:
                state_dict['pupiltrace']=np.append(state_dict['pupiltrace'],
                          np.ones([pcount,rlen-plen])*np.nan,
                          axis=1)

            # generate pupil signals
            t_pupil = nems.signal.RasterizedSignal(
                    fs=options['rasterfs'], data=state_dict['pupiltrace'],
                    name='pupil', recording=cellid, chans=['pupil'],
                    epochs=event_times
                    )

            if i == 0:
                pupil = t_pupil
            else:
                pupil = pupil.concatenate_time([pupil, t_pupil])

        if options['stim']:
            t_stim = nb.dict_to_signal(stim_dict, fs=options['rasterfs'],
                                    event_times=event_times)
            t_stim.recording = cellid

            if i == 0:
                print("i={0} starting".format(i))
                stim = t_stim
            else:
                print("i={0} concatenating".format(i))
                stim = stim.concatenate_time([stim, t_stim])

        if options['stim'] and options["runclass"] == "RDT":
            t_BigStimMatrix = state_dict['BigStimMatrix']
            del state_dict['BigStimMatrix']

            t_stim1 = nb.dict_to_signal(
                    stim1_dict, fs=options['rasterfs'],
                    event_times=event_times, signal_name='stim1',
                    recording_name=cellid
                    )
            t_stim2 = nb.dict_to_signal(
                    stim2_dict, fs=options['rasterfs'],
                    event_times=event_times, signal_name='stim2',
                    recording_name=cellid
                    )
            t_state = nb.dict_to_signal(
                    state_dict, fs=options['rasterfs'],
                    event_times=event_times, signal_name='state',
                    recording_name=cellid
                    )
            t_state.chans = ['repeating_phase', 'single_stream', 'targetid']

            if i == 0:
                print("i={0} starting".format(i))
                stim1 = t_stim1
                stim2 = t_stim2
                state = t_state
                BigStimMatrix = t_BigStimMatrix
            else:
                print("i={0} concatenating".format(i))
                stim1 = stim1.concatenate_time([stim1, t_stim1])
                stim2 = stim2.concatenate_time([stim2, t_stim2])
                state = state.concatenate_time([state, t_state])
                BigStimMatrix = np.concatenate(
                        (BigStimMatrix, t_BigStimMatrix), axis=2
                        )

    resp.meta = options

    signals = {'resp': resp}

    if options['pupil']:
        signals['pupil'] = pupil
    if options['stim']:
        signals['stim'] = stim

    if options['stim'] and (options["runclass"] == "RDT"):
        signals['stim1'] = stim1
        signals['stim2'] = stim2
    if options["runclass"] == "RDT":
        signals['state'] = state
        # signals['stim'].meta = {'BigStimMatrix': BigStimMatrix}

    rec = nems.recording.Recording(signals=signals)
    return rec


# things are breaking in nb.baphy_load_dataset(), event_times only contain trial information
# and not any o the stimulus relevant information in exptevents, here I redefine the method
# with a couple of changes to preserve these tims
def baphy_load_dataset_mateo(parmfilepath, options):
    # get the relatively un-pre-processed data
    exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams = \
        nb.baphy_load_data(parmfilepath, options)

    # pre-process event list (event_times) to only contain useful events
    # extract each trial
    print('Creating trial events')
    tag_mask_start = "TRIALSTART"
    tag_mask_stop = "TRIALSTOP"
    ffstart = (exptevents['name'] == tag_mask_start)
    ffstop = (exptevents['name'] == tag_mask_stop)
    TrialCount = np.max(exptevents.loc[ffstart, 'Trial'])
    event_times = pd.concat([exptevents.loc[ffstart, ['start']].reset_index(),
                             exptevents.loc[ffstop, ['end']].reset_index()],
                            axis=1)
    event_times['name'] = "TRIAL"
    event_times = event_times.drop(columns=['index'])

    # figure out length of entire experiment
    file_start_time = np.min(event_times['start'])
    file_stop_time = np.max(event_times['end'])

    # add event characterizing outcome of each behavioral
    # trial (if behavior)
    print('Creating trial outcome events')
    note_map = {'OUTCOME,FALSEALARM': 'FA_TRIAL',
                'OUTCOME,MISS': 'MISS_TRIAL',
                'BEHAVIOR,PUMPON,Pump': 'HIT_TRIAL'}
    this_event_times = event_times.copy()
    any_behavior = False
    for trialidx in range(1, TrialCount+1):
        ff = (((exptevents['name'] == 'OUTCOME,FALSEALARM')
              | (exptevents['name'] == 'OUTCOME,MISS')
              | (exptevents['name'] == 'BEHAVIOR,PUMPON,Pump'))
              & (exptevents['Trial'] == trialidx))

        for i, d in exptevents.loc[ff].iterrows():
            # print("{0}: {1} - {2} - {3}"
            #       .format(i, d['Trial'], d['name'], d['end']))
            this_event_times.loc[trialidx-1, 'name'] = note_map[d['name']]
            any_behavior = True

    te = pd.DataFrame(index=[0], columns=(event_times.columns))
    if any_behavior:
        # only concatenate newly labeled trials if events occured that reflect
        # behavior. There's probably a less kludgy way of checking for this
        # before actually running through the above loop
        event_times = pd.concat([event_times, this_event_times])
        te.loc[0] = [file_start_time, file_stop_time, 'ACTIVE_EXPERIMENT']
    else:
        te.loc[0] = [file_start_time, file_stop_time, 'PASSIVE_EXPERIMENT']
    event_times = event_times.append(te)

    # remove events DURING or AFTER LICK
    print('Removing post-response stimuli')
    ff = (exptevents['name'] == 'LICK')
    keepevents = np.ones(len(exptevents)) == 1
    for i, d in exptevents.loc[ff].iterrows():
        trialidx = d['Trial']
        start = d['start']
        fflate = ((exptevents['end'] > start)
                  & (exptevents['Trial'] == trialidx)
                  & (exptevents['name'].str.contains('Stim , ')))
        for i, d in exptevents.loc[fflate].iterrows():
            # print("{0}: {1} - {2} - {3}>{4}"
            #       .format(i, d['Trial'], d['name'], d['end'], start))
            # remove Pre- and PostStimSilence as well
            keepevents[(i-1):(i+2)] = False

    print("Keeping {0}/{1} events that precede responses"
          .format(np.sum(keepevents), len(keepevents)))
    exptevents = exptevents[keepevents].reset_index()

    # ff = (exptevents['Trial'] == 3)
    # exptevents.loc[ff]

    stim_dict = {}

    if 'pertrial' in options and options['pertrial']:
        # NOT COMPLETE!

        # make stimulus events unique to each trial
        this_event_times = event_times.copy()
        for eventidx in range(0, TrialCount):
            event_name = "TRIAL{0}".format(eventidx)
            this_event_times.loc[eventidx, 'name'] = event_name
            if options['stim']:
                stim_dict[event_name] = stim[:, :, eventidx]
        event_times = pd.concat([event_times, this_event_times])

    else:
        # generate stimulus events unique to each distinct stimulus
        ff_tar_events = exptevents['name'].str.contains('Target')
        ff_pre_all = exptevents['name'] == ""
        ff_post_all = ff_pre_all.copy()
        ff_sound_all = exptevents['name'] == ""

        # this is the place to hijack the code and add the oddball stimulus event paradimg
        # infers the names of the stimulus form the experiment parameters
        # and uses them to replace tags, which are not working for SSA.
        ff_tone_evetns = exptevents['name'].str.contains('Stim')
        freq_ids = tuple(exptparams['TrialObject'][1]['ReferenceHandle'][1]['Frequencies'])
        freq_rates = tuple(exptparams['TrialObject'][1]['ReferenceHandle'][1]['F1Rates'])
        freq_rates = ['{0:.2f}'.format(ff) for ff in freq_rates]
        freq_rates.append('ONSET')
        #creates stim tags based on frequencies and rates used
        stim_tags = ['{}+{}'.format(ii,jj) for ii, jj in itt.product(freq_ids, freq_rates) ]
        oldtags = copy.copy(tags)
        tags = stim_tags

        for eventidx in range(0, len(tags)):

            if options['stim']:
                # save stimulus for this event as separate dictionary entry
                stim_dict["STIM_" + tags[eventidx]] = stim[:, :, eventidx]
            else:
                stim_dict["STIM_" + tags[eventidx]] = np.array([[]])
            # complicated experiment-specific part
            tag_mask_start = (
                    "PreStimSilence , " + tags[eventidx] + " , Reference"
                    )
            tag_mask_stop = (
                    "PostStimSilence , " + tags[eventidx] + " , Reference"
                    )
            tag_mask_sound = ("Stim , " + tags[eventidx] + " , Reference")

            ffstart = (exptevents['name'] == tag_mask_start)
            if np.sum(ffstart) > 0:
                ffstop = (exptevents['name'] == tag_mask_stop)
                ffsound = (exptevents['name'] == tag_mask_sound)
            else:
                ffstart = (exptevents['name'].str.contains(tag_mask_start))
                ffstop = (exptevents['name'].str.contains(tag_mask_stop))
                ffsound = (exptevents['name'].str.contains(tag_mask_sound))

            # create intial list of stimulus events (including pre and post stim silences)
            this_event_times = pd.concat(
                    [exptevents.loc[ffstart, ['start']].reset_index(),
                     exptevents.loc[ffstop, ['end']].reset_index()],
                    axis=1
                    )
            this_event_times = this_event_times.drop(columns=['index'])
            this_event_times['name'] = "STIM_" + tags[eventidx]

            # screen for conflicts with target events
            keepevents = np.ones(len(this_event_times)) == 1
            for i, d in this_event_times.iterrows():
                f = (ff_tar_events
                     & (exptevents['start'] < d['end']-0.001)
                     & (exptevents['end'] > d['start']+0.001))

                if np.sum(f):
                    # print("Stim (event {0}: {1:.2f}-{2:.2f} {3}"
                    #       .format(eventidx,d['start'], d['end'],d['name']))
                    # print("??? But did it happen?"
                    #       "? Conflicting target: {0}-{1} {2}"
                    #       .format(exptevents['start'][j],
                    #               exptevents['end'][j],
                    #               exptevents['name'][j]))
                    keepevents[i] = False

            if np.sum(keepevents == False):
                print("Removed {0}/{1} events that overlap with target"
                      .format(np.sum(keepevents == False), len(keepevents)))

            # create final list of these stimulus events
            this_event_times = this_event_times[keepevents]
            tff, = np.where(ffstart)
            ffstart[tff[keepevents == False]] = False
            tff, = np.where(ffstop)
            ffstop[tff[keepevents == False]] = False

            event_times = event_times.append(this_event_times, ignore_index=True)
            this_event_times['name'] = "REFERENCE"
            event_times = event_times.append(this_event_times, ignore_index=True)
            # event_times = pd.concat([event_times, this_event_times])

            ff_pre_all = ff_pre_all | ffstart
            ff_post_all = ff_post_all | ffstop
            ff_sound_all = ff_sound_all | ffsound

        # generate list of corresponding pre/post events
        this_event_times2 = pd.concat(
                [exptevents.loc[ff_pre_all,['start']],
                 exptevents.loc[ff_pre_all,['end']]],
                axis=1
                )
        this_event_times2['name'] = 'PreStimSilence'
        this_event_times3 = pd.concat(
                [exptevents.loc[ff_post_all, ['start']],
                 exptevents.loc[ff_post_all,['end']]],
                axis=1
                )
        this_event_times3['name'] = 'PostStimSilence'
        # generate list of corresponding sound events
        this_event_times4 = pd.concat(
            [exptevents.loc[ff_sound_all, ['start']],
             exptevents.loc[ff_sound_all, ['end']]],
            axis=1
        )
        this_event_times4['name'] = 'Sound'

        event_times = event_times.append(this_event_times2, ignore_index=True)
        event_times = event_times.append(this_event_times3, ignore_index=True)
        event_times = event_times.append(this_event_times4, ignore_index=True)
        # event_times = pd.concat(
        #         [event_times, this_event_times2, this_event_times3]
        #         )

    # add behavior events
    if exptparams['runclass'] == 'PTD' and any_behavior:
        # special events for tone in noise task
        tar_idx_freq = exptparams['TrialObject'][1]['TargetIdxFreq']
        tar_snr = exptparams['TrialObject'][1]['RelativeTarRefdB']
        common_tar_idx, = np.where(tar_idx_freq == np.max(tar_idx_freq))

        if (isinstance(tar_idx_freq, (int))
                or len(tar_idx_freq) == 1
                or np.isinf(tar_snr[0])):
            diff_event = 'PURETONE_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (np.max(common_tar_idx) < 2):
            diff_event = 'EASY_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (2 in common_tar_idx):
            diff_event = 'MEDIUM_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (np.min(common_tar_idx) > 2):
            diff_event = 'HARD_BEHAVIOR'
        else:
            diff_event = 'PURETONE_BEHAVIOR'
        te = pd.DataFrame(index=[0], columns=(event_times.columns))
        te.loc[0] = [file_start_time, file_stop_time, diff_event]
        event_times = event_times.append(te, ignore_index=True)
        # event_times=pd.concat([event_times, te])

    # sort by when the event occured in experiment time
    event_times = event_times.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    event_times = event_times.drop(columns=['index'])

    return event_times, spike_dict, stim_dict, state_dict

rec_mateo= baphy_load_recording_mateo(cellid,batch,options)