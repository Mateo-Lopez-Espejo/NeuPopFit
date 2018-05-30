#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import io

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils

#import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw

import logging
log = logging.getLogger(__name__)


cellid = 'chn002h-a1'
batch = 296
modelname = 'env100pt_dlog_fir2x15_lvl1_dexp1_basic'

modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'


autoPlot = True
saveInDB = False

log.info('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(
        cellid, batch))

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

# generate xfspec, which defines sequence of events to load data,
# generate modelspec, fit data, plot results and save
recording_uri = nw.generate_recording_uri(cellid, batch, loader)
#recording_uri = "/Users/svd/python/nems/recordings/TAR010c-06-1.tgz"

# generate xfspec, which defines sequence of events to load data,
# generate modelspec, fit data, plot results and save
# xfspec = nw.generate_loader_xfspec(cellid,batch,loader)
xfspec = xhelp.generate_loader_xfspec(loader, recording_uri)

# xfspec.append(['nems.initializers.from_keywords_as_list',
#               {'keyword_string': modelspecname, 'meta': meta},
#               [],['modelspecs']])
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

# xfspec += nw.generate_fitter_xfspec(cellid, batch, fitkey)
xfspec += xhelp.generate_fitter_xfspec(fitkey)

xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

if autoPlot:
    # GENERATE PLOTS
    xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
# ctx, log_xf = xforms.evaluate(xfspec)
# Evaluate the xforms

# Create a log stream set to the debug level; add it as a root log handler
log_stream = io.StringIO()
ch = logging.StreamHandler(log_stream)
ch.setLevel(logging.DEBUG)
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
ch.setFormatter(formatter)
rootlogger = logging.getLogger()
rootlogger.addHandler(ch)

ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

# Close the log, remove the handler, and add the 'log' string to context
log.info('Done (re-)evaluating xforms.')
ch.close()
rootlogger.removeFilter(ch)

log_xf = log_stream.getvalue()

modelspecs = ctx['modelspecs']
val = ctx['val'][0]

# save some extra metadata
destination = '/auto/data/tmp/modelspecs/{0}/{1}/{2}/'.format(
        batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
modelspecs[0][0]['meta']['modelpath'] = destination
modelspecs[0][0]['meta']['figurefile'] = destination + 'figure.0000.png'

# save results
log.info('Saving modelspec(s) to {0} ...'.format(destination))
xforms.save_analysis(destination,
                     recording=ctx['rec'],
                     modelspecs=modelspecs,
                     xfspec=xfspec,
                     figures=ctx['figures'],
                     log=log_xf)

if saveInDB:
    # save in database as well
    nd.update_results_table(modelspecs[0])
