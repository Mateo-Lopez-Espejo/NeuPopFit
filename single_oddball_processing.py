import os
import io
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems_db.db as nd
import logging
import joblib as jl


def single_oddball_processing(cellid, batch, modelname, force_refit=False, save_in_DB=False):
    '''

    full a oddball analisis: loads data
                             strim as rasterized point process i.e. edge or onset
                             set oddball dependent epochs
                             set jitter status epochs

                             short term plasticity
                             fir
                             dc gain shift
                             fit

                             calculates ssa index for [actual, predicted] * [f1, f2, cell]
                             calculates activiti level   ''        ''       ''  ''   ''


    :param cellid: str of cell id
    :param batch: batch number under stephen convention, default is SSA batch 296
    :param modelname: str defining the modules comprising the model. ToDo Should exclude the loading key!
    :param force_refit: Bool. if true fits the model regardless of cached values, replaces cached values
    :param save_in_DB: ??? TODO what is this doing
    :return: experimetn context ctx. contains recordings and modelspecs
    '''

    # cellid = 'gus037d-a1'
    # batch = 296
    # modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'

    log = logging.getLogger(__name__)
    batch = batch

    log.info('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(
        cellid, batch))

    # parse modelname
    kws = modelname.split("_")
    # ToDo, is this good practice with the loader? it should be included in modelspec
    loader = 'OddballLoader'
    modelspecname = "_".join(kws[0:-1])
    fitkey = kws[-1]

    # figure out some meta data to save in the model spec
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    # recording_uri = nw.generate_recording_uri(cellid, batch, loader)

    # generate xfspec, which defines sequence of events to load data,
    # generate modelspec, fit data, plot results and save
    xfspec = list()

    # loader
    xfspec.append(['oddball_xforms.load_oddball',
                   {'cellid': cellid}])

    # give oddball format: stim as rasterized point process, nan as zeros, oddball epochs, jitter status epochs,
    xfspec.append(['oddball_xforms.give_oddball_format', {'scaling': 'same'}])

    # define model architecture
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])

    # adds jackknife, fitter and prediction
    xfspec.extend(xhelp.generate_fitter_xfspec(fitkey))

    # TODO add a xform to cache and pull from cache when possible

    # add metrics correlation
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # add SSA related metrics
    # val, modelspecs, sub_epoch, super_epoch, baseline
    jitters = ['Jitter_On', 'Jitter_Off', 'Jitter_Both']
    xfspec.append(['oddball_xforms.calculate_oddball_metrics',
                   {'sub_epoch': 'Stim', 'super_epoch': jitters, 'baseline': 'silence'},
                   ['val', 'modelspecs'], ['modelspecs']])

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

    # # TODO, get rid of this temporal caching. temporal ctx caching
    # if force_refit is False and os.path.exists(
    #         '/home/mateo/oddball_analysis/pickles/180601_test_oddball_fit_file_path'):
    #     print('using cached ctx')
    #     ctx = jl.load('/home/mateo/oddball_analysis/pickles/180601_test_oddball_fit_file_path')
    #     xfspec = xfspec[5:]

    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)
        # for caches the fitted parameters for the sake of speed
        # if xfa[0] == 'nems.xforms.fit_nfold':
        #     jl.dump(ctx, '/home/mateo/oddball_analysis/pickles/180601_test_oddball_fit_file_path')


    # caches the fitted models TODO implement somehow

    # Close the log, remove the handler, and add the 'log' string to context
    log.info('Done (re-)evaluating xforms.')
    ch.close()
    rootlogger.removeFilter(ch)

    log_xf = log_stream.getvalue()

    modelspecs = ctx['modelspecs']

    # save some extra metadata
    destination = '/auto/users/mateo/oddball_metrics_results/{0}/{1}/{2}/'.format(
        batch, cellid, modelname)
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

    if save_in_DB:
        # save in database as well TODO why is saving as single elemetn of modelspec?
        nd.update_results_table(modelspecs[0], preview=None,
                                username="MLE", labgroup="lbhb")

    return ctx


'''
testing suit

cellid = 'gus037d-a2'
modelname = 'stp2_fir2x15_lvl1_basic-nftrial'
single_oddball_processing(cellid=cellid, modelname=modelname, force_refit=False, save_in_DB = False)
'''
