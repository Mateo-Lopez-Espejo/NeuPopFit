import nems.xforms as xforms
import nems.xform_helper as xhelp
import oddball_functions as of


# TODO two xfroms. one transforming stimulus into rasterized point process and another to do the SI and activity metrics calculation


# extracted from xforms.evaluate_step
'''
Helper function for evaluate. Take one step
SVD revised 2018-03-23 so specialized xforms wrapper functions not required
but now xfa can be len 4, where xfa[2] indicates context in keys and
xfa[3] is context out keys
'''

def stim_as_rasterized_point_process(rec, **context):
    rec = of.as_rasterized_point_proces(rec, scaling='same')
    return {'rec': rec}

def calculate_oddball_metrics(rec, **context):
    SSA_index = of.get_recording_SI(rec, scaling='same')
    activity_lvl = of.get_recording_activity(rec)
    modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs, rec=rec)
    if False:
        return {'rec': rec, }
    return {'modelspecs': modelspecs}
