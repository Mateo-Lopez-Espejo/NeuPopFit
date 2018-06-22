import oddball_functions as of

def stim_as_rasterized_point_process(rec, scaling, **context):
    # rasterizes all signal
    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    rec = of.as_rasterized_point_process(rec, scaling=scaling)
    return {'rec': rec}

def calculate_oddball_metrics(val, modelspecs, sub_epoch, baseline, **context):
    # this thing should get the specific recordings from which i wanna calculate SI and activity of the validation
    # dataset

    # the val in the ctx is a list of recordings, probably each of the validation subsets when doing jackknife multiple
    # evaluatio validation
    # TODO work the data flow for jackknifed data.
    val = val[0]

    # update modelspecs with the adecuate metadata,
    SI = of.get_recording_SI(val, sub_epoch)
    modelspecs[0][0]['meta']['SSA_index'] = SI
    RA = of.get_recording_activity(val, sub_epoch, baseline=baseline)
    modelspecs[0][0]['meta']['activity'] = RA

    return modelspecs


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
