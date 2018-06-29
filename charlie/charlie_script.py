import nems.preprocessing as preproc
import nems.modelspec as ms
import nems
import nems_db.db as nd
import os
import sys

sys.path.append('/auto/users/hellerc/code/charlie_nems_tools')
sys.path.append('/auto/users/hellerc/code/my_scripts/state_dependent_noise_correlations/')
import tools as ts
import coupling_model_tools as cmt
import loading_utilities as lu

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    print("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

cell = sys.argv[1]
batch = sys.argv[2]
modelname = sys.argv[3]  # fs-min_isolation_state1_state2_state3_...

site = cell[:-5]

modelstring = modelname.split('*')[0]  # type of analysis/fit being done

state_sigs = []
for i, sig in enumerate(modelname.split('_')):
    if i == 0:
        rec_load_vars = sig.split('*')[1]
    else:
        state_sigs.append(sig)

fs = int(rec_load_vars.split('-')[0])
min_isolation = int(rec_load_vars.split('-')[1])
batch = int(batch)
recache = False

rawid = ts.which_rawids(batch=batch, site=site)

recording = lu.load_recording(batch=int(batch),
                              site=site,
                              fs=fs,
                              min_isolation=min_isolation,
                              rawid=rawid,
                              recache=recache)

# Preprocess for fitting (cut out invalid segments and create psth estimate)
if int(batch) == 307:
    newrec = recording.create_mask()
    newrec = newrec.or_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT'])
    newrec = newrec.and_mask(['REFERENCE'])

else:
    newrec = recording.create_mask()
    newrec = recording.or_mask(['REFERENCE'])

if 'subset' in modelstring:
    # case for NAT sounds, exclude sounds that weren't played more than 9 times
    if batch == 289:
        stims = (newrec.epochs['name'].value_counts() > 9)
        stims = [stims.index[i] for i, s in enumerate(stims) if 'STIM' in stims.index[i] and s == True]
        newrec = newrec.and_mask(stims)
        print('masking only stims: {0}'.format(stims))

# Add psth prediction to the recordings for all ests and vals
print('calculating psth prediction...')
newrec = preproc.generate_psth_from_resp(newrec)

# state signals are created within this function
pred, modelspec = cmt.fit_single_state_gain(newrec=newrec,
                                            cell=cell,
                                            state_sigs=state_sigs)

# Save modelspecs
filepath = '/auto/users/hellerc/noise_correlation_results/'

filepath += str(batch) + '/'

if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath += cell + '/'

if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath += modelname

print('saving modelspecs...')
ms.save_modelspec(modelspec, filepath + '.json')
print('saved modelspecs')

print('fitting shuffled models...')
for shuf in state_sigs:
    pred, modelspec = cmt.fit_single_state_gain(newrec=newrec,
                                                cell=cell,
                                                state_sigs=state_sigs,
                                                shuffle=shuf)
    f = filepath + '_shuffle_' + shuf
    ms.save_modelspec(modelspec, f + '.json')
    print('completed {0} shuffle fit'.format(shuf))

if queueid:
    nd.update_job_complete(queueid)