import nems_db.baphy as nb
import nems.xforms as xforms
import nems_db.xform_wrappers as nw
import os
"""
there are two options to load oddball data, yet there are slight differences in the resultin loaded recording 
"""


cellid = 'gus037d-a1' # cell with both jitter status (regular and random intervals), and 2 stim resp discrepancies
#cellid = 'gus037d-a2' # cell with only one jitter status (regular intervals?), and 1 stim resp discrepancies
batch = 296

# this loads a cell using nems_db.baphy.baphy_load_recording(cellid, batch, **options)


options = {}
options["stimfmt"] = "envelope"
options["chancount"] = 0
options["rasterfs"] = 100
options['includeprestim'] = 1
options['runclass'] = 'SSA'
rec = nb.baphy_load_recording_nonrasterized(cellid, batch, **options)
rec['resp'] = rec['resp'].rasterize()
rec['stim'] = rec['stim'].rasterize()
baphy_load = rec

# thise loads the same cell, batchs using the xform you gave me a while ago in the example
# ultimately it calls:
# nems.recording.load_recording_from_targz_stream('/auto/data/nems_db/recordings/296/envelope0_fs100/gus037d-a2.tgz')

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

ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

xform_load = ctx['rec']
xform_load['resp'] = xform_load['resp'].rasterize()
xform_load['stim'] = xform_load['stim'].rasterize()

# there are four main discrepancies
# 1. time: xform_load is faster as is already cached, however i don't know what prepossessing it has
# 2. NaNs: baphy_load has the nans in the 'stim' signals that we already have talked about. easy to solve, already done.
# the previous are minor issues
# the following are more incovenient
# 3. size: in baphy_load 'resp' signals have a couple more data points than 'stim' signals, in this example 1. This
# 4. baphy_load signals have epochs correspondign to the file name wich allows me to pull the file parameters and tell
#    what jitter status is in what parts of the recording.


recs = {'baphy': baphy_load, 'xform': xform_load}

for key, val in recs.items():
    print('{}\n'
          'stim shape: {}\n'
          'resp shape: {}'.format(key, val['stim'].shape, val['resp'].shape))


for key, val in recs.items():
    # file in epochs: e.g. 'FILE_gus037d03_p_SSA'
    print('{}\n'
          'epochs: {}'.format(key, val['stim'].epochs.name.unique()))



