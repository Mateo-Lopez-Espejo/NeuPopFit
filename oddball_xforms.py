import oddball_functions as of
import oddball_db as od
import nems_db.baphy as nb
import nems.recording as recording


'''
This is here Just for reference to help me figure out how xfomrms are working
xfa in my case will have 4 possitions:
0.    is the name of the function to run, in my particular case
      the above funtions e.g. 'oddball_xforms.stim_as_rasterized_point_process'
1.    a dictionary with the arguments for the xform function. in the above example, no arguments {}
      in the case of 'oddball_xforms.calculate_oddball_metrics' probably {'sub_epoch': ['Stim', 'PostStimSilence']}
2.    list of keys to pull elements from ctx and use as arguments in the function defined ien 0. in the case of
      'oddball_xforms.calculate_oddball_metrics' ['val', 'modelspecs'] as the validation subsed used to calculate the
      metrics and the modelspecs dictionary to store such values
3.    list of key for the output arguments. number o keys must be equal to number of function outputs. This keys are used
      to create new o replace existing values in the ctx input dictionary.
'''

def stim_as_rasterized_point_process(rec, scaling, **context):
    # rasterizes all signal
    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    rec = of.as_rasterized_point_process(rec, scaling=scaling)
    return {'rec': rec}

def calculate_oddball_metrics(val, modelspecs, sub_epoch, super_epoch, baseline, **context):
    # calculates SSA index and activity index for the validatio. asumes unique validation recording.
    if len(val) != 1:
        raise Warning("multiple val recordings, usign the first. Inverse jackknife if recomended before this step")

    val = val[0]

    valid_jitters = {'Jitter_Off', 'Jitter_On', 'Jitter_Both'}
    if not set(super_epoch).issubset(valid_jitters):
        raise ValueError("super_epoch must be a subset of {}".format(valid_jitters))

    # update modelspecs with the adecuate metadata, calculates SI and activity for each super_epoch
    modelspecs[0][0]['meta']['SSA_index'] = dict.fromkeys(super_epoch)
    modelspecs[0][0]['meta']['activity'] = dict.fromkeys(super_epoch)

    for sup_ep in super_epoch:
        dict_key = sup_ep
        if sup_ep == 'Jitter_Both':
            sup_ep = None

        SI = of.get_recording_SI(val, sub_epoch, super_epoch=sup_ep)
        modelspecs[0][0]['meta']['SSA_index'][dict_key] = SI
        RA = of.get_recording_activity(val, sub_epoch, super_epoch='Jitter_Off', baseline=baseline)
        modelspecs[0][0]['meta']['activity'][dict_key] = RA
    # todo test this beauty
    return modelspecs

def give_oddball_format(rec, scaling, **context):
    '''
    a bunch of functions formating signals within the recording for later oddball analysis.
    '''
    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    # sets stim as onset with amplitu ecual as maximum value of original stim
    rec = of.as_rasterized_point_process(rec, scaling=scaling)
    # changes Nan values in to zero for the 'stim' signal
    rec = of.recording_nan_as_zero(rec, ['stim'])
    # set epochs of Jitter On and Jitter Off
    rec = of.set_recording_jitter_epochs(rec)
    # set oddball epochs
    rec = of.set_recording_oddball_epochs(rec)
    return {'rec': rec}

def load_oddball(cellid, recache=False, **context):
    cellid = cellid
    batch = 296
    options = {}
    options['recache'] = recache
    options['stimfmt'] = "envelope"
    options['chancount'] = 0
    options['rasterfs'] = 100
    options['includeprestim'] = 1
    options['runclass'] = 'SSA'
    rec_path = nb.baphy_data_path(cellid, batch, **options) # gets the path, if it does not exists, loads and caches.
    rec = recording.load_recording(rec_path)
    return {'rec': rec}


