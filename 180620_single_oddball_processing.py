import matplotlib.pyplot as plt
import numpy as np
import os
import io
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils
import nems_db.db as nd
import nems_db.xform_wrappers as nw
import logging
import oddball_xforms

def single_oddball_processing():
    log = logging.getLogger(__name__)

    cellid = 'gus037d-a2'
    batch = 296

    modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    autoPlot = False
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

    recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    # generate xfspec, which defines sequence of events to load data,
    # generate modelspec, fit data, plot results and save
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
    xfspec += xhelp.generate_fitter_xfspec(fitkey)

    # add metrics correlation
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # add SSA related metrics
    xfspec.append(['oddball_xforms.calculate_oddball_metrics', {'sub_epoch': 'Stim', 'baseline': 'silence'},
                   ['val', 'modelspecs'], ['modelspecs']])

    if autoPlot:
        # GENERATE PLOTS
        xfspec.append(['nems.xforms.plot_summary',    {}])


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

    return modelspecs


specs = single_oddball_processing()