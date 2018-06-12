# TODO two xfroms. one transforming stimulus into rasterized point process and another to do the SI and activity metrics calculation

import NeuPopFit.oddball_functions as of


def stim_as_rasterized_point_process(rec, **context):
    rec = of.as_rasterized_point_proces(rec, scaling='same')
    return {'rec': rec}

def calculate_oddball_metrics(rec, **context):


    SSA_index = of.get_recording_SI(rec, scaling='same')
    activity_lvl = of.get_recording_activity(rec)

    return {'rec': rec, }

    modelspecs = nems.analysis.api.standard_correlation(
        est, val, modelspecs, rec=rec)

    return {'modelspecs': modelspecs}
