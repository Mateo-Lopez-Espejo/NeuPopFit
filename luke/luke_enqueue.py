#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:20:54 2018

@author: luke
"""

import nems_db.db as nd

# python environment where you want to run the job
executable_path='/auto/users/luke/miniconda3/envs/nemsenv/bin/python'

# name of script that you'd like to run
#script_path='/auto/users/luke/Projects/SPS/NEMS/fit_single_SPO.py'
script_path='/auto/users/luke/Code/nems_db/nems_queue_luke.py'

# parameters that will be passed to script as argv[1], argv[2], argv[3]:
parm1='fre196b-08-1_1417-351'   # for nems_fit_single, this is cellid
parm2='306'    # for nems_fit_single, this is the batch #

parm3='env100_dlog_fir2x15_lvl1_fit01'  # for nems_fit_single, this is the modelname
parm3='env100_dlog_fir2x15_lvl1_dexp1_fit01'  # for nems_fit_single, this is the modelname
parm3='env100_dlog_stp2_fir2x15_lvl1_dexp1_fit01'  # for nems_fit_single, this is the modelname


force_rerun = True   # true if job already has been run and you want to rerun
user = 'luke'            # will be used for load balancing across users

pcellids=nd.get_batch_cells(306)
pcellids=pcellids['cellid'].tolist()

for pcellid in pcellids:
    print('Add {}'.format(pcellid))
    qid,msg=nd.enqueue_single_model(pcellid, parm2, parm3, force_rerun=force_rerun,
                        executable_path=executable_path,
                        script_path=script_path, user=user)