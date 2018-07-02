import nems_db.db as nd

# python environment where you want to run the job
executable_path='/auto/users/mateo/miniconda3/envs/nemsenv/bin/python'

# name of script that you'd like to run
#script_path='/auto/users/luke/Projects/SPS/NEMS/fit_single_SPO.py'
script_path='/auto/users/mateo/oddball_analysis/cluster_script_180629.py'

# parameters that will be passed to script as argv[1], argv[2], argv[3]:
parm1 = cellid = 'gus037d-a2'
parm2 = batch = 296 
parm3 = modelname = 'stp2_fir2x15_lvl1_basic-nftrial' 

user = 'Mateo'
force_rerun = True



# if i want to fir all the cells in a bathc i have to iterate over them before calling the enqueue:
# for cell in batch:
#     enqueueue_single_model(cell, ...)

qid,msg = nd.enqueue_single_model(cellid=cellid, batch=batch, modelname=modelname,
                                  user=user, session=None, force_rerun=force_rerun,
                                  executable_path=executable_path, script_path=script_path)
