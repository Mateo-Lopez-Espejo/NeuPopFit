import nems_db.db as nd
cellid = 'gus037d-a2'
batch = 296
modelname = 'env100pt_stp2_fir2x15_lvl1_basic-nftrial'
user = 'Mateo'
force_rerun = True
script_path = '/auto/users/mateo/oddball_analysis/single_oddball_processing.py'

nd.enqueue_single_model(cellid=cellid, batch=batch, modelname=modelname, user=user,
                        session=None,
                        force_rerun=force_rerun, codeHash="master", jerbQuery='',
                        executable_path=None, script_path=script_path)
