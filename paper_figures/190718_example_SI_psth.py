import pandas as pd
import numpy as np
import joblib as jl
import oddball_DF as odf
import scipy.stats as sst
import seaborn as sns
import matplotlib.pyplot as plt
import oddball_plot as op
import os


# this block for the linear vs wc-stp
modelname1 = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname1 = 'LN STRF'
modelname2 = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname2 = 'RW-STP STRF'

modelnames = [modelname1, modelname2]
shortnames = [shortname1, shortname2]




example_cell = 'chn066b-c1'
op.cell_psth(example_cell, modelnames[0])
# op.cell_psth(example_cell, modelnames[1])


