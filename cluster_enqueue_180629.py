import nems_db.db as nd
import itertools as itt

# python environment where you want to run the job
executable_path='/auto/users/mateo/miniconda3/envs/nemsenv/bin/python'
# name of script that you'd like to run
script_path='/auto/users/mateo/oddball_analysis/cluster_script_180629.py'


# parameters that will be passed to script.
force_rerun = True
parm2 = batch = 296

# define cellids
batch_cells = nd.get_batch_cells(batch=296).cellid
#batch_cells = ['gus037d-a1']
batch_cells = ['gus019d-b1', 'gus019d-b2', 'gus019e-a1',
            'gus019e-b1', 'gus020c-a1', 'gus020c-c1',
            'gus021c-a1', 'gus021c-b1', 'gus021f-a1',
            'gus021f-a2', 'gus035b-c1']




# define modelname
loaders = ['odd']
ests = vals = ['jof', 'jon']
modelnames = ['{}_fir2x15_lvl1_basic-nftrial_est-{}_val-{}'.format(loader, est, val) for
              loader, est, val in itt.product(loaders, ests, vals)]


modelnames = ['odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal',
              'odd.1_stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal',
              'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal']


# only old cells without jitter status
#batch_cells = [cellid for cellid in batch_cells if cellid[0:3] != 'gus']

for cellid, modelname in itt.product(batch_cells, modelnames):
    qid,msg = nd.enqueue_single_model(cellid=cellid, batch=batch, modelname=modelname,
                                  user='Mateo', session=None, force_rerun=force_rerun,
                                  executable_path=executable_path, script_path=script_path)

    print(msg)