import pandas as pd
import nems.modelspec as ms
import nems_db.db as nd
import nems_db.baphy as nb
import os
import joblib as jl

# important db root path
this_script_dir = os.path.dirname(os.path.realpath(__file__))
pickles_path = '{}/pickles'.format(this_script_dir)

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

def get_single_metadata(cellid, batch):
    '''
    returns cell file metadata, like experiment parameters, right now only gets jitter values
    :param cellid:
    :param batch:
    :return:
    '''

    # check what files are asociated to this cellid batch combination

    parmfiles = nd.get_batch_cell_data(cellid=cellid, batch=296)
    parmfiles = list(parmfiles['parm'])
    # creates a dictionary mapping the epoch keys to the parmfiles paths, i.e.
    # from: 'FILE_gus037d03_p_SSA' to: '/auto/data/daq/Augustus/gus037/gus037d03_p_SSA.m'
    parmfiles = ['{}.m'.format(path) for path in parmfiles]


    # ToDo add here whatever metatdata is important to extract and tabulate
    cell_meta = dict.fromkeys(['cellid', 'jitter_status'])
    cell_meta['cellid'] = cellid

    # to relate filename to jitter status pulls experimetn parameters
    jitter_status = list()
    for pfile in parmfiles:
        globalparams, exptparams, exptevents = nb.baphy_parm_read(pfile)

        # convoluted indexig into the nested dictionaries ot get Jitter status, sets value to "Off" by default
        j_stat = exptparams['TrialObject'][1]['ReferenceHandle'][1].get('Jitter', 'Off')
        jitter_status.append('Jitter_{}'.format(j_stat.rstrip()))

    cell_meta['jitter_status'] = jitter_status

    df = pd.DataFrame([cell_meta])

    return df

def get_batch_metadata(batch, recache=False):
    '''
    returns a DF wit metadata of all the cells in a batch. this metadata is relevanta at the moment of decidign what
    modelse to fit to the cells.
    In the case of

    :param batch:
    :return:
    '''

    filename = 'batch_meta'
    meta_db_path = os.path.normcase('{}/{}'.
                                    format(pickles_path, filename))

    if os.path.exists(meta_db_path) and recache == False:
        print('df loaded from cache ')
        DF = jl.load(meta_db_path)
        return DF

    else:
        print('extracting meta de novo')

    batch_cells = nd.get_batch_cells(batch=296).cellid

    metas = list()

    for cellid in batch_cells:
        df = get_single_metadata(cellid, batch)
        metas.append(df)

    DF = pd.concat(metas)

    jl.dump(DF, meta_db_path)

    return DF






