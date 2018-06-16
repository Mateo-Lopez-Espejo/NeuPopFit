import matplotlib.pyplot as plt
import joblib as jl
import numpy as np
import os
import io

import oddball_functions as of
import oddball_xforms as ox

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils

import nems_db.db as nd
import nems_db.xform_wrappers as nw
'''
set of testing functions, load pickled test data set, passes through, function in oddball_functions
it is highly recomended doing al real testing as functions within this file, for more efficient debuging using pycharm
debuger.... Lets hope I can keep, my word.
'''

# test files. the paths will be different between my desktop and laptop.
test_ctx_file_path = '/home/mateo/NeuPopFit/pickles/180531_test_context_full'
test_load_file_path = '/home/mateo/NeuPopFit/pickles/180601_test_context_only_load'

def ctx():
    ctx = test_ctx = jl.load(test_ctx_file_path)
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
    odd_sig = set_signal_oddball_epochs(sig)
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
