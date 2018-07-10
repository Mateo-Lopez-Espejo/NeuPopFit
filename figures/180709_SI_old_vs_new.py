import oddball_test as ot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl

old = jl.load('/home/mateo/batch_296/171115_all_subset_fit_eval_combinations_DF')
new = jl.load('/home/mateo/oddball_analysis/pickles/180709_DF_all_parms_only_jal')
cellids = old.cellid.unique().tolist()

# old filter
modelname = 'env100e_fir20_fit01_ssa'
modelname = 'env100e_stp1pc_fir20_fit01_ssa'

ff_model = old.model_name == modelname
ff_param = old.parameter == 'SI'
ff_stream = old.stream == 'cell'
ff_jitter = old.Jitter =='On'
ff_actpred = old.act_pred == 'predicted'

oldfilt = old.loc[ff_model & ff_param & ff_stream & ff_jitter & ff_actpred, :].drop_duplicates(subset=['cellid'])
print(oldfilt.duplicated(['cellid']).any())

oldfilt.set_index('cellid', inplace=True)
oldfilt=oldfilt.rename(columns = {'values':'old_value'})


# new filter
modelname = 'odd_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'
modelname = 'odd_stp2_fir2x15_lvl1_basic-nftrial_est-jal_val-jal'

ff_model = new.modelname == modelname
ff_param = new.parameter == 'SSA_index'
ff_stream = new.stream == 'cell'
ff_jitter = new.Jitter =='Jitter_On'
ff_actpred = new.act_pred == 'pred'
ff_cell  = new.cellid.isin(cellids)

newfilt = new.loc[ff_model & ff_param & ff_stream & ff_jitter & ff_actpred & ff_cell, :].drop_duplicates(subset=['cellid'])
print(newfilt.duplicated(['cellid', 'value']).any())

newfilt.set_index('cellid', inplace=True)
newfilt=newfilt.rename(columns = {'value': 'new_value'})


DF = pd.concat([newfilt['new_value'], oldfilt['old_value']], axis=1, sort=True)

DF = DF.astype(float)

DF.plot('old_value', 'new_value', kind='scatter')

