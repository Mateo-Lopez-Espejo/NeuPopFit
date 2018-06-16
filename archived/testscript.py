import nems.initializers
import nems.preprocessing as preproc
import nems.analysis.api
import nems.utils
from nems.fitters.api import scipy_minimize
from archived import import_ssa as isa
import joblib as jl

#%% Defines file values and import
# parameters for PSTH + pupil analysis. stim==false to skip loading it
options = {'rasterfs': 20,
           'stim': False,
           'includeprestim': True,
           'stimfmt': 'none',
           'chancount': 0}

# specify a subset of cells to load

# need to pass one cell id for nems_db to search for parameter file in
# celldb
cellid = 'gus037d-a2'
batch = 296

modelstring = "stategain2"  # one-dim state variable plus a constant offset dim TODO: change to old model

# load recording wont work traditionally because stim is saved as an envelope
print('Loading recording...')
rec = isa.baphy_load_recording_mateo(cellid, batch, options)
#rec = nb.baphy_load_recording_nonrasterized(cellid, batch, options)

#%% pickle the recording for later offline usage
### this was a recording object with a single RasterizedSignal object inside.
jl_path = '/home/mateo/NeuPopFit/180420_ssa_rec'
jl.dump = (rec, jl_path)

#%% import recording from local pickle
jl_path = 'path' # path to the pickle
rec = jl.load(jl_path)

#%% Gets a state signal from the recording, e.g. pupil. I dont know how pupil is being saved
print('Generating state signal...')
rec = preproc.make_state_signal(rec, ['pupil'], [''], 'state')

#%% INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

print('Initializing modelspec(s)...')
meta = {'cellid': cellid, 'batch': batch, 'modelstring': modelstring,
        'modelname': modelstring}
modelspec = nems.initializers.from_keywords(modelstring, meta=meta)
modelspecs = [modelspec]

#%% DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.


print('Setting up jackknifes...')
# create all jackknife sets
nfolds = 10
ests, vals, m = preproc.split_est_val_for_jackknife(rec, modelspecs=None,
                                                    njacks=nfolds)

# generate PSTH prediction for each set
ests, vals = preproc.generate_psth_from_est_for_both_est_and_val_nfold(ests,
                                                                       vals)
#%% RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

print('Fitting modelspec(s)...')
modelspecs_out = nems.analysis.api.fit_nfold(
        ests, modelspecs, fitter=scipy_minimize)
modelspecs = modelspecs_out