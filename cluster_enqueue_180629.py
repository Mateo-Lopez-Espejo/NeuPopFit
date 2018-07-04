import nems_db.db as nd

# python environment where you want to run the job
executable_path='/auto/users/mateo/miniconda3/envs/nemsenv/bin/python'

# name of script that you'd like to run
script_path='/auto/users/mateo/oddball_analysis/cluster_script_180629.py'

# parameters that will be passed to script as argv[1], argv[2], argv[3]:
parm1 = cellid = 'gus037d-a1' # overwriten later, this was for thest
parm2 = batch = 296 
parm3 = modelname = 'stp2_fir2x15_lvl1_basic-nftrial' 

user = 'Mateo'
force_rerun = True


batch_cells = nd.get_batch_cells(batch=296).cellid
# only old cells without jitter status
#batch_cells = [cellid for cellid in batch_cells if cellid[0:3] != 'gus']

for cellid in batch_cells:
    qid,msg = nd.enqueue_single_model(cellid=cellid, batch=batch, modelname=modelname,
                                  user=user, session=None, force_rerun=force_rerun,
                                  executable_path=executable_path, script_path=script_path)