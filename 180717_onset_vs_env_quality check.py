import matplotlib.pyplot as plt
import os
import io
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems_db.db as nd
import logging
import oddball_xforms as ox
import numpy as np
import oddball_plot as op

cellid = 'gus037d-a1'
cellid = 'chn066b-c1'
batch = 296
modelnames = ['odd_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal',
              'odd1_stp2_fir2x15_lvl1_basic-nftrial_si-jk_est-jal_val-jal']


as_point_process = [True, False]
offset = 0.5
length = 1000

fig, axes = plt.subplots(1,2)
axes = np.ravel(axes)

for app, ax in zip(as_point_process, axes):

    # loads
    ctx = ox.load_oddball(cellid, recache=False)

    # holds to plot
    stim1 = ctx['rec']['stim'].as_continuous().T
    toplot1 = stim1[:length, :]

    # give oddball format: stim as rasterized point process, nan as zeros, oddball epochs, jitter status epochs,
    ctx = ox.give_oddball_format(scaling='same', as_point_process=app, **ctx)

    # holds to plot
    stim2 = ctx['rec']['stim'].as_continuous().T
    toplot2 = stim2[:length, :]

    ax.plot(toplot1, label='original')
    ax.plot(toplot2, label='modified')
#
#
# for modelname in modelnames:
#     op.cell_psth(cellid, modelname)
