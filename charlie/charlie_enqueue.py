import nems_db.db as nd
from itertools import permutations
import re
import os
import numpy as np
import sys

sys.path.append('/auto/users/hellerc/code/my_scripts/state_dependent_noise_correlations/')
import tools as ts


def make_modelname(fs, min_isolation, state_sigs, modelstring):
    modelname = '-'.join([str(fs), str(min_isolation)])
    s = '_'.join(state_sigs)
    modelname += '_' + s
    modelname = modelstring + '*' + modelname
    return modelname


batch = 289  # 307, 294 (pup voc), 289 (NAT/pup)
fs = 5
min_isolation = 70
# state_sigs = ['coupling', 'pupil', 'pupcoupling', 'active', 'behcoupling']
# state_sigs = ['r1', 'r2', 'pupil'] #, 'behavior']
state_sigs = ['pupil']
modelstring = 'singlestategainsubset'  # 'pupgaincorrectionrscsw20'  #pairwise, rscsw20
modelname = [make_modelname(fs, min_isolation, state_sigs, modelstring)]
filepath = '/auto/users/hellerc/noise_correlation_results/' + str(batch) + '/'  # for permissions reasons

sites = ['TAR017b', 'bbl104h', 'bbl099g', 'BRT034f', 'BRT033b', 'BRT032e', 'BRT026c']
# 307: ['TAR010c', 'BRT026c', 'bbl102d', 'BRT033b'], 289: ['BOL005c', 'BOL006b']

script = '/auto/users/hellerc/code/my_scripts/state_dependent_noise_correlations/' \
         'nems_fitting_functions/fit_single_state_gain.py'

for site in sites:

    rawid = ts.which_rawids(batch=batch, site=site)

    cids = nd.get_stable_batch_cellids(batch=batch, cellid='%', rawid=rawid)
    mask = np.zeros(len(cids))
    for i, cid in enumerate(cids):
        if nd.get_isolation(cellid=cid, batch=batch).values > min_isolation:
            mask[i] = 1

    cids = list(cids[list(mask == 1)])

    # pairs = list(permutations(cids,2))
    # cids = [re.sub("['() ]", '', str(i)) for i in pairs]
    os.chmod(filepath, 0o777)
    nd.enqueue_models(celllist=cids,
                      batch=batch,
                      modellist=modelname,
                      executable_path='/auto/users/nems/anaconda3/bin/python',
                      script_path=script,
                      user='hellerc')