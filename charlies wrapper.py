import nems_db.db as nd
import nems_db.baphy as nb
import nems_db.xform_wrappers as nw
import numpy as np
import sys
import os
from NEMS.nems.recording import Recording


def load_recording(batch=None, site=None, fs=None, min_isolation=None, options=None):
    if batch is None:
        sys.exit('Must specify batch')

    if site is None:
        sys.exit('Must specify site id')

    if fs is None:
        fs = 100
        print('WARNING: setting fs to default 100Hz')

    if min_isolation is None:
        min_isolation = 75
        print('WARNING: setting min_isolation to default 75%')

    if options is None:
        print('WARNING: using default options for baphy_load_recording')
        options = {}
        options["stimfmt"] = "ozgf"
        options["chancount"] = 18
        options["rasterfs"] = fs
        options['includeprestim'] = 1
        options['stim'] = False
        options['pupil'] = True
        options['pupil_deblink'] = True
        options['pupil_median'] = 1
        # options["average_stim"]=True
        # options["state_vars"]=[]

    if options['rasterfs'] != fs:
        options['rasterfs'] = fs  # check to make sure these agree (for filenaming sake)

    # Get the correct cellids
    print('Querying database to find all cellids...')
    cids = nd.get_stable_batch_cellids(batch=batch, cellid=site)
    mask = np.zeros(len(cids))
    for i, cid in enumerate(cids):
        if nd.get_isolation(cellid=cid, batch=batch).values > min_isolation:
            mask[i] = 1

    cids_correct = list(np.array(cids)[list(mask == 1)])

    # Tell options which cells to load
    options['cellid'] = cids_correct

    cachefilename = name_recording(batch, site, fs, min_isolation)
    cachepath = '/auto/users/hellerc/code/my_scripts/cachedRecordings/'
    # Load the data
    if not os.path.isfile(cachepath + cachefilename):
        print('loading recording from baphy using baphy_load_recording...')
        rec = nb.baphy_load_recording(cids_correct[0], batch, options)
        print('saving recording to cachedRecordings...')
        rec.save(cachepath + cachefilename)
    else:
        print('loading recording from cached zip file...')
        rec = Recording.load(cachepath + cachefilename)

    return (rec)


def name_recording(batch, site, fs, min_isolation):
    return ('batch' + str(batch) + '_site' + site + '_fs' + str(fs) + '_isolation' + str(min_isolation) + '.tar.gz')