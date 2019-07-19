import numpy as np
import matplotlib.pyplot as plt
import oddball_plot as op
import oddball_DB as odb
import scikits.bootstrap as bts
import oddball_functions as of
import pathlib as pl

"""

Works with older versions of NEMS (githash: 3a25cc5259f30e2b7a961e4a9fac2477e57b8144)
and nems_db (githash: 3fefdb537b100c346486266c97f18e3f55cb5086)


"""

# this block for the linear vs wc-stp
modelname1 = 'odd.1_fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname1 = 'LN STRF'
modelname2 = 'odd.1_wc.2x2.c-stp.2-fir.2x15-lvl.1_basic-nftrial_si.jk-est.jal-val.jal'
shortname2 = 'RW-STP STRF'

modelnames = [modelname1, modelname2]
shortnames = [shortname1, shortname2]

color1 = '#FDBF76'  # yellow for linear model
color2 = '#5054a5' # blue for CW STP


example_cell = 'chn066b-c1'

# order the response and predictions from the two models in a handy dictionary

signals = dict()
for mm, (model, shorname) in enumerate(zip(modelnames, shortnames)):
    ctx = odb.load_single_ctx(example_cell, batch=296, modelname=model)
    if mm == 0:
        signals['response'] = ctx['val'][0]['resp']

    signals[shorname] = ctx['val'][0]['pred']

#op.psth(ctx)

fs = 100
PostStim = PreStim = 0.15 # in seconds
Stim = 0.1 # in s


fig, ax = plt.subplots()

colors = ['black', color1, color2]
rates = [['std'], ['onset', 'dev']]
linestyles = ['-', ':']
rate_names = ['standard', 'deviant']

start = 10 # in bins
end = 30 # in bin

for (sig_name, signal), color in zip(signals.items(),colors):

    folded_sig = of.extract_signal_oddball_epochs(signal, sub_epoch=None, super_epoch=None)

    for rate, rate_name, linestyle in zip(rates, rate_names, linestyles):

        array = [val for key, val in folded_sig.items() if key.split('_')[1] in rate]
        array = np.concatenate(array, axis=0)

        array = array * fs # transforms into e Hz
        array = array[:,:,start:end]



        ci = np.asarray([bts.ci(array[:, 0, tt], np.mean, n_samples=100, method='pi') for tt in range(array.shape[2])])
        psth = np.mean(array, axis=0).squeeze()

        t = np.linspace(-PreStim, Stim + PostStim, int(fs*(PreStim+Stim+PostStim)))
        t = t[start:end]
        ax.plot(t, psth, color=color, linestyle=linestyle, linewidth=3, alpha=0.8, label=f'{sig_name} {rate_name}')
        ax.fill_between(t, ci[:, 0], ci[:, 1], color=color, alpha=0.2)

ax.axvline(0, color='black')
ax.axvline(Stim, color='black')

ax.set_ylabel('spike rate (Hz)')
ax.set_xlabel('time (s)')
ax.legend(loc='upper left', fontsize='large')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=15)
ax.title.set_size(20)
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

# set figure to full size in tenrec screen
fig.set_size_inches(19.2, 9.79)


root = pl.Path(f'/home/mateo/Pictures/STP_paper')
filename = f'SI_prediction_eg'
if not root.exists(): root.mkdir(parents=True, exist_ok=True)

png = root.joinpath(filename).with_suffix('.png')
fig.savefig(png, transparent=True, dpi=100)

svg = root.joinpath(filename).with_suffix('.svg')
fig.savefig(svg, transparent=True)


