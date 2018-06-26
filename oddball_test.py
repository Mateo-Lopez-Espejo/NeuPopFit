import matplotlib.pyplot as plt
import joblib as jl
import numpy as np
import os
import io

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils
import nems_db.db as db
import nems_db.xform_wrappers as nw
import nems_db.baphy as nb  # baphy-specific functions

import oddball_functions as of
import oddball_xforms as ox

# reload modules just in case??
import imp

imp.reload(of)
imp.reload(ox)

'''
set of testing functions, load pickled test data set, passes through, function in oddball_functions
it is highly recomended doing al real testing as functions within this file, for more efficient debuging using pycharm
debuger.... Lets hope I can keep, my word.
'''

# test files. the paths will be different between my desktop and laptop.
test_ctx_file_path = '/home/mateo/NeuPopFit/pickles/180531_test_context_full'
test_load_file_path = '/home/mateo/NeuPopFit/pickles/180601_test_context_only_load'


def ctx():
    ctx = jl.load(test_ctx_file_path)
    return ctx


def rec():
    ctx = test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    return rec


def sig():
    ctx = test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    sig = rec['resp']
    return sig


def old_df():
    filename = '/home/mateo/batch_296/171117_6model_all_eval_DF'
    DF = jl.load(filename)
    return DF


def odd_ctx():
    filename = '/home/mateo/NeuPopFit/pickles/180621_test_oddball_ctx'
    ctx = jl.load(filename)
    return ctx


def newcells():
    cellids = ['chn002h-a1', 'chn002h-a2', 'chn004b-a1', 'chn004c-b1', 'chn005d-a1', 'chn005d-c1', 'chn006a-b1',
               'chn008a-c1', 'chn008a-c2', 'chn008b-a1', 'chn008b-b1', 'chn008b-c1', 'chn008b-c2', 'chn008b-c3',
               'chn009b-a1', 'chn009d-a1', 'chn010c-a1', 'chn010c-a2', 'chn010c-c3', 'chn012d-a1', 'chn016b-d1',
               'chn016c-c1', 'chn016c-d1', 'chn019a-c1', 'chn019a-d1', 'chn019a-d2', 'chn020b-b1', 'chn020f-b1',
               'chn022c-a1', 'chn022e-a1', 'chn022e-a2', 'chn023c-d1', 'chn023e-d1', 'chn029d-a1', 'chn030b-c1',
               'chn030f-a1', 'chn030f-c1', 'chn062c-c1', 'chn062f-a2', 'chn063b-d1', 'chn063d-d1', 'chn063h-b1',
               'chn065c-d1', 'chn065d-c1', 'chn066b-c1', 'chn066c-a1', 'chn066d-a1', 'chn067c-b1', 'chn067d-b1',
               'chn068d-d1', 'chn069b-d1', 'chn069c-b1', 'chn073b-b1', 'chn073b-b2', 'chn079b-d1', 'chn079d-b1',
               'eno001f-a1', 'eno002c-c1', 'eno002c-c2', 'eno005d-a1', 'eno006c-c1', 'eno006d-c1', 'eno007b-a1',
               'eno008c-b1', 'eno008e-b1', 'eno009c-b1', 'eno009d-a1', 'eno009d-a2', 'eno010f-b1', 'eno013d-a1',
               'eno013d-a2', 'eno013d-a3', 'eno035c-a1', 'gus016c-a1', 'gus016c-c1', 'gus016c-c2', 'gus019c-a1',
               'gus019c-b1', 'gus019c-b2', 'gus019c-b3', 'gus019d-b1', 'gus019d-b2', 'gus019e-a1', 'gus019e-b1',
               'gus020c-a1', 'gus020c-c1', 'gus021c-a1', 'gus021c-b1', 'gus021f-a1', 'gus021f-a2', 'gus022b-a1',
               'gus023e-c1', 'gus023f-c1', 'gus023f-d1', 'gus025b-a1', 'gus025b-c1', 'gus026c-a3', 'gus026d-a1',
               'gus030d-b1', 'gus035a-a1', 'gus035a-a2', 'gus035b-c1', 'gus036b-b1', 'gus036b-c1', 'gus036b-c2',
               'gus037d-a1', 'gus037d-a2', 'gus037e-d1', 'gus037e-d2']
    return cellids


def oldcells():
    filename = '/home/mateo/batch_296/171117_6model_all_eval_DF'
    DF = jl.load(filename)
    cellids = DF.cellid.unique().tolist()
    return cellids


def set_signal_oddball_epochs():
    test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    sig = rec['resp']
    odd_sig = of.set_signal_oddball_epochs(sig)
    print(odd_sig.epochs.head(20))
    return odd_sig


def get_superepoch_subset():
    ctx = test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    sig = rec['resp']
    sig.add_epoch('test_epoch_1', np.array([[0, 101]]))
    sig.add_epoch('test_epoch_2', np.array([[100, 201]]))
    sig.add_epoch('test_epoch_3', np.array([[200, 300]]))
    epochs = sig.epochs
    newepochs =of.get_superepoch_subset(sig, ['test_epoch_1', 'test_epoch_2', 'test_epoch_3'])
    return epochs, newepochs


def get_signal_SI():
    test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    sig = rec['resp']
    SI = of.get_signal_SI(sig, None)
    return SI


def get_signal_activity():
    test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    sig = rec['resp']
    act = of.get_signal_activity(sig, None)


def as_rasterized_point_process():
    # not sure what this is doing, dont care anymore...
    loaded_ctx = jl.load(test_load_file_path)
    fig, ax = plt.subplots()
    val = loaded_ctx['val']
    stim = val.get_signal('stim')
    ax.plot(stim.as_continuous()[0, :])
    of.as_rasterized_point_process(val)
    stim = val.get_signal('stim')
    ax.plot(stim.as_continuous()[0, :])


def preprocess():
    # tests a string of xfspecs containing some default loading xpforms and custom made oddball_xpforms
    # so far this is working perfectly

    cellid = 'gus037d-a2'
    batch = 296
    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # finds raw data location
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    xfspec = list()

    # loader
    recordings = [recording_uri]
    normalize = False
    xfspec.append(['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}])

    # stim as point process
    xfspec.append(['oddball_xforms.stim_as_rasterized_point_process', {'scaling': 'same'}])

    # estimation validation subsets
    xfspec.append(['nems.xforms.use_all_data_for_est_and_val',
                   {}])

    ctx = {}
    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)

    return ctx


def xform_load():
    # tests a string of xfspecs containing some default loading xpforms and custom made oddball_xpforms

    cellid = 'gus016c-a2'  # this cell is not working for some reason
    cellid = 'gus037d-a2' # this cell is not in the old list of good cells, but it works
    batch = 296
    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # finds raw data location
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    xfspec = list()

    # loader
    recordings = [recording_uri]
    normalize = False
    xfspec.append(['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}])
    # this is ultimately calling nems.recording.load_recording_from_targz(recording_uri)

    ctx = {}
    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)

    return ctx


def baphy_load():
    cellid = 'gus037d-a2'
    batch = 296
    options = {}
    options["stimfmt"] = "envelope"
    options["chancount"] = 0
    options["rasterfs"] = 100
    options['includeprestim'] = 1
    options['runclass'] = 'SSA'
    rec = nb.baphy_load_recording(cellid, batch, **options)
    return rec


def oddball_full_analysis():
    ''' this test just run the whole analisis, with custom preprocesing, model fit, and post procecing metrics calculations'''
    cellid = 'gus016c-a2'  # this cell is not working for some reason
    cellid = 'gus037d-a2'  # this cell only has jitter on for some reason? another cell in the same recording session has both jitters
    cellid = 'gus037d-a1'  # this cell is not in the old list of good cells, but it works
    batch = 296
    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # finds raw data location
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    xfspec = list()

    # loader
    recordings = [recording_uri]
    normalize = False
    xfspec.append(['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}])

    # stim as point process
    xfspec.append(['oddball_xforms.stim_as_rasterized_point_process', {'scaling': 'same'}])

    # define model architecture
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])

    '''
    # add jackknife
    tfolds = 5
    xfspec.append(['nems.xforms.mask_for_jackknife',
                   {'njacks': tfolds, 'epoch_name': 'TRIAL'}])

    # add fitter
    xfspec.append(['nems.xforms.fit_nfold', {}])

    # add predic
    xfspec.append(['nems.xforms.predict', {}])
    '''

    # adds jackknife, fitter and prediction
    xfspec.extend(xhelp.generate_fitter_xfspec(fitkey))

    # add metrics correlation
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # add SSA related metrics
    xfspec.append(['oddball_xforms.calculate_oddball_metrics', {'sub_epoch': 'Stim', 'baseline': 'silence'},
                   ['val', 'modelspecs'], ['modelspecs']])
    ctx = {}
    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)

    return ctx


def SI_metrics():
    ctx = jl.load(test_ctx_file_path)

    xfa = ['oddball_xforms.calculate_oddball_metrics', {'sub_epoch': 'Stim', 'baseline': 'silence'},
           ['val', 'modelspecs'], ['modelspecs']]

    ctx = xforms.evaluate_step(xfa, ctx)

    return ctx


def check_SI_error():
    '''
    there was an error ein SI calculation, turns out the predicted signal has long chuncks of nan at the begining of each
    trial, which encompass part of the onset values. when calculating the PSTHs of these onsets, the nans raise errors
    of "mean of empty slice".
    this revealed the inconsistency between the nan for resp, pred and the mask of each signal.
    '''
    # extract pre fitted model, it already has SI related indices
    filename = '/home/mateo/NeuPopFit/pickles/180621_test_oddball_ctx'
    ctx = jl.load(filename)
    val = ctx['val'][0]

    sub_epoch = 'Stim'
    baseline = 'silence'

    # re calculate the SI related indices
    SI = of.get_recording_SI(val, sub_epoch)
    RA = of.get_recording_activity(val, sub_epoch, baseline=baseline)

    return SI, RA


def loader_error():
    cellid = 'gus016c-a2'  # this cell is not working for some reason
    # cellid = 'gus037d-a2' # this cell is not in the old list of good cells, but it works
    batch = 296
    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # finds raw data location
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)
    return recording_uri


def nan_issue():
    '''
    Suddenly for some reason the issue is no longer there
    '''
    cellid = 'gus016c-a2'  # this cell is not working for some reason
    cellid = 'gus037d-a2'  # this cell is not in the old list of good cells, but it works
    batch = 296
    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # finds raw data location
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    xfspec = list()

    # loader
    recordings = [recording_uri]
    normalize = False
    xfspec.append(['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}])

    # stim as point process
    xfspec.append(['oddball_xforms.stim_as_rasterized_point_process', {'scaling': 'same'}])

    # estimation validation subsets, This is redundant given the jackknife generating the est val subsets
    xfspec.append(['nems.xforms.use_all_data_for_est_and_val',
                   {}])

    # define model architecture
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])

    # adds jackknife, fitter and prediction
    xfspec.extend(xhelp.generate_fitter_xfspec(fitkey))

    # add metrics correlation
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # add SSA related metrics
    xfspec.append(['oddball_xforms.calculate_oddball_metrics', {'sub_epoch': 'Stim', 'baseline': 'silence'},
                   ['val', 'modelspecs'], ['modelspecs']])

    ctx = {}
    ctx_ls = list()
    fig, axes = plt.subplots(len(xfspec), 3)
    for row, xfa in enumerate(xfspec):
        new_ctx = xforms.evaluate_step(xfa, ctx)
        ctx_ls.append(new_ctx)
        ctx = new_ctx
        # get the response of the estimation and validation data set,
        step_name = xfa[0]

        for col, [recname, rec] in enumerate(ctx.items()):
            if recname not in ['rec', 'est', 'val']:
                continue
            else:
                pass

            if isinstance(rec, list):
                pass
            else:
                # makes into a list
                rec = [rec]

            # iterates over each recordign, takes the signal, gets the nans, plots

            for ii, thisrec in enumerate(rec):
                sig = thisrec['stim']
                respnan = np.isnan(sig.rasterize().as_continuous().T)
                ax = axes[row, col]
                ax.plot(respnan, label='jackknife {}'.format(ii))
                if col == 0:
                    ax.set_ylabel(step_name)
                ax.set_title(recname)
    return ctx, ctx_ls


def oddball_format():
    # load the file
    cellid = 'gus037d-a1'
    batch = 296
    options = {}
    options["stimfmt"] = "envelope"
    options["chancount"] = 0
    options["rasterfs"] = 100
    options['includeprestim'] = 1
    options['runclass'] = 'SSA'
    rec = nb.baphy_load_recording(cellid, batch, **options)

    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    # sets stim as onset with amplitu ecual as maximum value of original stim
    rec = of.as_rasterized_point_process(rec, scaling='same')
    # changes Nan values in to zero for the 'stim' signal
    rec = of.recording_nan_as_zero(rec, ['stim'])
    # set epochs of Jitter On and Jitter Off
    rec = of.set_recording_jitter_epochs(rec)
    # set oddball epochs
    rec = of.set_recording_oddball_epochs(rec)
    return {'rec': rec}

def all_custom_xforms():
    cellid = 'gus037d-a1'  # this cell is not in the old list of good cells, but it works
    batch = 296
    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # finds raw data location
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    xfspec = list()

    # loader
    xfspec.append(['oddball_xforms.load_oddball',
                   {'cellid': cellid}])

    # give oddball format: stim as rasterized point process, nan as zeros, oddball epochs, jitter status epochs,
    xfspec.append(['oddball_xforms.stim_as_rasterized_point_process', {'scaling': 'same'}])

    # define model architecture
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])


    # adds jackknife, fitter and prediction
    xfspec.extend(xhelp.generate_fitter_xfspec(fitkey))

    # add metrics correlation
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # add SSA related metrics
    # val, modelspecs, sub_epoch, super_epoch, baseline
    jitters = ['Jitter_On', 'Jitter_Off', 'Jitter_Both']
    xfspec.append(['oddball_xforms.calculate_oddball_metrics', {'sub_epoch': 'Stim', 'super_epoch': jitters, 'baseline': 'silence'},
                   ['val', 'modelspecs'], ['modelspecs']])
    ctx = {}
    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)

    return ctx

# odd_ctx = all_custom_xforms()

