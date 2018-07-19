import pandas as pd
import nems.modelspec as ms
import nems_db.db as nd
import nems_db.baphy as nb
import os
import joblib as jl
import nems.xforms as xforms
import numpy as np
import itertools as itt

# for local pickling
this_script_dir = os.path.dirname(os.path.realpath(__file__))
pickles_path = '{}/pickles'.format(this_script_dir)


def get_source_dir(cellid, batch, modelname):
    source_dir = '/auto/users/mateo/oddball_results/{0}/{1}/{2}/'.format(
        batch, cellid, modelname)

    return source_dir


def load_single_ctx(cellid, batch, modelname):
    '''
    load the modelspec of an oddball recordign including SSA related metrics within the metadata
    :param cellid: str the name of the cell
    :param batch: num, for oddbal data it is 296
    :param modelname: str a string of characters describing the model
    :return: xfspec and context
    '''
    source_dir = get_source_dir(cellid, batch, modelname)
    xfspec, ctx = xforms.load_analysis(source_dir, eval_model=True)

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


def get_modelnames():
    # ToDo dont forget to keep adding modelspecs
    # ToDo make it to import a list of all the modelpairs... somehow

    modelnames = list()
    # null and alternative models with onset as envelope
    modelnames.append('odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal')
    modelnames.append('odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal')

    # different fit eval jitter subsets
    loaders = ['odd']
    ests = vals = ['jof', 'jon']
    modelnames.extend(['{}_stp2_fir2x15_lvl1_basic-nftrial_est-{}_val-{}'.format(loader, est, val)
                       for loader, est, val in itt.product(loaders, ests, vals)])

    # null and alternative models with stim as envelope
    modelnames.append('odd1_fir2x15_lvl1_basic-nftrial_est-jal_val-jal')
    modelnames.append('odd1_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal')

    # same as previous but with ssa index calculated over jackknifes
    modelnames.append('odd_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal')
    modelnames.append('odd_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal')
    modelnames.append('odd1_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal')
    modelnames.append('odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal')

    # model with crosstalk between channels, partly following the new keyword paradigm
    modelnames.append('odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal')


    modelnames = np.asarray(modelnames)

    return modelnames

def get_wierd_cells():
    wierd_cells = ['gus019d-b1', 'gus019e-a1', 'gus019e-b1', 'gus020c-a1', 'gus020c-c1', 'gus021c-a1', 'gus021c-b1',
                   'gus021f-a1', 'gus021f-a2', 'gus035b-c1']

    return wierd_cells







