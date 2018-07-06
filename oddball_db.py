import nems.modelspec as ms
import os
import joblib as jl

# important db root path
root_path = '/auto/users/mateo/oddball_results'

# less important test cache path
cache_path = '/home/mateo/oddball_analysis/pickles'


def save_oddball_modelspecs(ctx):
    '''
    saves modeslpecs to my personal file of (right now for testing).
    :param ctx: the context object related with a chain of xpform operations
    :return: None TOdo, should return somethign for later use?
    '''

    # parses important metadata and modelspecs
    modelspecs = ctx['modelspecs']
    meta = modelspecs[0][0]['meta']
    cellid = meta['cellid']
    batch = meta['batch']
    modelname = meta['modelname']

    # definese directory based on my personal file/batch/cellid/modelname, crates if non existent
    filepath = root_path
    filepath = '{}/{}'.format(filepath, str(batch))
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    filepath = '{}/{}'.format(filepath, cellid)

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    filepath = '{}/{}'.format(filepath, modelname)


    print('saving modelspecs at {} ...'.format(filepath))
    ms.save_modelspec(modelspecs, filepath + '_test.json')
    print('saved modelspecs')

    return None

def cache_fitted_context(ctx):
    '''
    creates a temporal cache with the entire final_ctx already fitted, so refitting is not required
    for calculation of postprocessing metrics

    :param ctx: the context output of a chain of xpform operations
    :return: none
    '''
    modelspecs = ctx['modelspecs']
    meta = modelspecs[0][0]['meta']
    cellid = meta['cellid']
    batch = meta['batch']
    modelname = meta['modelname']

    # definese directory based on my personal file/batch/cellid/modelname, crates if non existent
    filepath = cache_path
    filepath = '{}/{}'.format(filepath, str(batch))
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    filepath = '{}/{}'.format(filepath, cellid)

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    filepath = '{}/{}'.format(filepath, modelname)

    print('saving final_ctx...')
    jl.dump(ctx, filepath)
    print('saved final_ctx')

    return None

def load_from_cache(cellid, batch, modelname):
    '''

    Pulls a cached context including recordigns and fitting.

    :param cellid: str e.g. 'gus037d-a1'
    :param batch: int, svc batch systems e.g. 296 for ssa
    :param modelanem: str. chain of model arguments e.g. 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'
    :return: final_ctx, dict. contains recordings and modelspecs
    '''

    filepath = '{}/{}/{}/{}'.format(cache_path ,batch, cellid, modelname)
    ctx = jl.load(filepath)

    return ctx
