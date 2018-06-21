import matplotlib.pyplot as plt
import joblib as jl
import numpy as np
import os
import io

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils
import nems_db.db as nd
import nems_db.xform_wrappers as nw

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


def set_signal_oddball_epochs():
    test_ctx = jl.load(test_ctx_file_path)
    rec = test_ctx['val'][0]
    sig = rec['resp']
    odd_sig = of.set_signal_oddball_epochs(sig)
    print(odd_sig.epochs.head(20))
    return odd_sig


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


def loader():
    # tests a string of xfspecs containing some default loading xpforms and custom made oddball_xpforms

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

    # estimation validation subsets
    xfspec.append(['nems.xforms.use_all_data_for_est_and_val',
               {}])

    ctx = {}
    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)


    return ctx


def oddball_full_analysis():
    ''' this test just run the whole analisis, with custom preprocesing, model fit, and post procecing metrics calculations'''
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

    # define model architecture
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])

    # add fitter
    xfspec.append(xhelp.generate_fitter_xfspec(fitkey)[0])

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

    xfa =['oddball_xforms.calculate_oddball_metrics', {'sub_epoch': 'Stim', 'baseline': 'silence'},
                   ['val', 'modelspecs'], ['modelspecs']]

    ctx = xforms.evaluate_step(xfa, ctx)

    return ctx

ctx = oddball_full_analysis()
