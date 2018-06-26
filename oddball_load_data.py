import nems_db.baphy as bd
import nems_db.db as db
import numpy as np
import pandas as pd
import math
import re
import os


stim_cache_dir = '/auto/data/tmp/tstim/'  # location of cached stimuli
spk_subdir = 'sorted/'   # location of spk.mat files relative to parmfiles

def baphy_load_data(parmfilepath, **options):
    """
    this feeds into baphy_load_recording and baphy_load_recording_RDT (see
        below)
    input:
        parmfilepath: baphy parameter file
        options: dictionary of loading options
            runclass: matches Reference1 or Reference2 events, depending

    current outputs:
        exptevents: pandas dataframe with one row per event. times in sec
              since experiment began
        spiketimes: list of lists. outer list indicates unit, inner list is
              the set of spike times (secs since expt started) for that unit
        unit_names: list of strings uniquely identifier each units by
              channel-unitnum (CC-U). can append to siteid- to get cellid
        stim: [channel X time X event] stimulus (spectrogram) matrix
        tags: list of string identifiers associate with each stim event
              (can be used to find events in exptevents)

    other things that could be returned:
        globalparams, exptparams: dictionaries with expt metadata from baphy

    """
    # default_options={'rasterfs':100, 'includeprestim':True,
    #                  'stimfmt':'ozgf', 'chancount':18,
    #                  'cellid': 'all'}
    # options = options.update(default_options)

    # add .m extension if missing
    if parmfilepath[-2:] != ".m":
        parmfilepath += ".m"
    # load parameter file
    globalparams, exptparams, exptevents = bd.baphy_parm_read(parmfilepath)

    # figure out stimulus cachefile to load
    if 'stim' in options.keys() and options['stim']:
        stimfilepath = bd.baphy_stim_cachefile(exptparams, parmfilepath, **options)
        print("Cached stim: {0}".format(stimfilepath))
        # load stimulus spectrogram
        stim, tags, stimparam = bd.baphy_load_specgram(stimfilepath)

        if options["stimfmt"]=='envelope' and \
            exptparams['TrialObject'][1]['ReferenceClass']=='SSA':
            # SSA special case
            stimo=stim.copy()
            maxval=np.max(np.reshape(stimo,[2,-1]),axis=1)
            print('special case for SSA stim!')
            ref=exptparams['TrialObject'][1]['ReferenceHandle'][1]
            stimlen=ref['PipDuration']+ref['PipInterval']
            stimbins=int(stimlen*options['rasterfs'])

            stim=np.zeros([2,stimbins,6])
            prebins=int(ref['PipInterval']/2*options['rasterfs'])
            durbins=int(ref['PipDuration']*options['rasterfs'])
            stim[0,prebins:(prebins+durbins),0:3]=maxval[0]
            stim[1,prebins:(prebins+durbins),3:]=maxval[1]
            tags=["{}+ONSET".format(ref['Frequencies'][0]),
                  "{}+{:.2f}".format(ref['Frequencies'][0],ref['F1Rates'][0]),
                  "{}+{:.2f}".format(ref['Frequencies'][0],ref['F1Rates'][1]),
                  "{}+ONSET".format(ref['Frequencies'][1]),
                  "{}+{:.2f}".format(ref['Frequencies'][1],ref['F1Rates'][0]),
                  "{}+{:.2f}".format(ref['Frequencies'][1],ref['F1Rates'][1])]

    else:
        stim = np.array([])

        if options['runclass'] is None:
            stim_object = 'ReferenceHandle'
        elif 'runclass' in exptparams.keys():
            runclass = exptparams['runclass'].split("_")
            if (len(runclass) > 1) and (runclass[1] == options["runclass"]):
                stim_object = 'TargetHandle'
            else:
                stim_object = 'ReferenceHandle'
        else:
            stim_object = 'ReferenceHandle'

        tags = exptparams['TrialObject'][1][stim_object][1]['Names']
        tags, tagids = np.unique(tags, return_index=True)
        stimparam = []

    # figure out spike file to load
    pp, bb = os.path.split(parmfilepath)
    spkfilepath = pp + '/' + spk_subdir + re.sub(r"\.m$", ".spk.mat", bb)
    print("Spike file: {0}".format(spkfilepath))

    # load spike times
    sortinfo, spikefs = bd.baphy_load_spike_data_raw(spkfilepath)

    # adjust spike and event times to be in seconds since experiment started
    exptevents, spiketimes, unit_names = bd.baphy_align_time(
            exptevents, sortinfo, spikefs, options['rasterfs']
            )

    # assign cellids to each unit
    siteid = globalparams['SiteID']
    unit_names = [(siteid + "-" + x) for x in unit_names]
    # print(unit_names)

    # test for special case where psuedo cellid suffix has been added to
    # cellid by stripping anything after a "_" underscore in the cellid (list)
    # provided
    pcellids = options['cellid'] if (type(options['cellid']) is list) \
       else [options['cellid']]
    cellids = []
    pcellidmap = {}
    for pcellid in pcellids:
        t = pcellid.split("_")
        t[0] = t[0].lower()
        cellids.append(t[0])
        pcellidmap[t[0]] = pcellid
    print(pcellidmap)
    # pull out a single cell if 'all' not specified
    spike_dict = {}
    for i, x in enumerate(unit_names):
        if (cellids[0] == 'all'):
            spike_dict[x] = spiketimes[i]
        elif (x.lower() in cellids):
            spike_dict[pcellidmap[x.lower()]] = spiketimes[i]

    if not spike_dict:
        raise ValueError('No matching cellid in baphy spike file')

    state_dict = {}
    if options['pupil']:
        try:
            pupilfilepath = re.sub(r"\.m$", ".pup.mat", parmfilepath)
            pupiltrace, ptrialidx = bd.baphy_load_pupil_trace(
                    pupilfilepath, exptevents, **options
                    )
            state_dict['pupiltrace'] = pupiltrace

        except:
            raise ValueError("Error loading pupil data: " + pupilfilepath)

    return (exptevents, stim, spike_dict, state_dict,
            tags, stimparam, exptparams)

def get_batch_params():
    # todo this is probably too slow and can be replaced with a smart sql querry...
    # load the full db
    parmfiles = db.get_batch_cell_data(batch=296, cellid=None, rawid=None, label=None)
    # resets indexes for ease??
    parmfiles = parmfiles.reset_index()

    #iterates over every row and gets relevant data from the parmfile and
    df = list()
    for index, row in parmfiles.iterrows():
        # row = next(parmfiles.iterrows())[1]
        this_row = dict.fromkeys(['rawid', 'cellid', 'parmfile'])
        this_row['rawid'] = row.rawid
        this_row['cellid'] = row.cellid
        this_row['parmfile'] = row.parm
        parms = db.get_data_parms(rawid=this_row['rawid'])
        # parses relevant data from the parms df
        relevant = ['Ref_Frequencies', 'Ref_F1Rates', 'Ref_Jitter']
        for rr in relevant:
            # for backwards compatibility, checks if Ref_Jitter exists, asumes off if not
            if rr == 'Ref_Jitter' and 'Ref_Jitter' not in parms.name.tolist():
                this_row[rr] = 'Off'
                continue
            val = parms.loc[parms.name==rr, ['value']].values[0]
            sval = parms.loc[parms.name==rr, ['svalue']].values[0][0]
            # checks if the parameter value has string or number value.
            if math.isnan(val) and sval is not None:
                vv = sval
            elif sval is None and not math.isnan(val):
                vv = val
            this_row[rr] = vv

        df.append(this_row)

    df =pd.DataFrame(df)

    return df


options = {'stimfmt': 'envelope', 'chancount': 0, 'rasterfs': 100, 'includeprestim': 1, 'runclass': 'SSA',
           'pertrial': 0, 'pupil': 0, 'pupil_deblink': 1, 'pupil_median': 1, 'stim': 1, 'cellid': 'gus037d-a2',
           'batch': 296, 'rawid': None}
parmfilepath = '/auto/data/daq/Augustus/gus037/gus037d04_p_SSA.m'
aaa = baphy_load_data(parmfilepath)