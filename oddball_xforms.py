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



# This is her Just for reference to help me figure out how xfomrms are working
def evaluate_step(xfa, context={}):
    '''
    Helper function for evaluate. Take one step
    SVD revised 2018-03-23 so specialized xforms wrapper functions not required
      but now xfa can be len 4, where xfa[2] indicates context in keys and
      xfa[3] is context out keys
    '''
    if not(len(xfa) == 2 or len(xfa) == 4):
        raise ValueError('Got non 2- or 4-tuple for xform: {}'.format(xfa))
    xf = xfa[0]
    xfargs = xfa[1]
    if len(xfa) > 2:
        context_in = {k: context[k] for k in xfa[2]}
    else:
        context_in = context
    if len(xfa) > 3:
        context_out_keys = xfa[3]
    else:
        context_out_keys = []

    fn = ms._lookup_fn_at(xf)
    # Check for collisions; more to avoid confusion than for correctness:
    for k in xfargs:
        if k in context_in:
            m = 'xf arg {} overlaps with context: {}'.format(k, xf)
            raise ValueError(m)
    # Merge args into context, and make a deepcopy so that mutation
    # inside xforms will not be propogated unless the arg is returned.
    merged_args = {**xfargs, **context_in}
    args = copy.deepcopy(merged_args)
    # Run the xf
    log.info('Evaluating: {}'.format(xf))
    new_context = fn(**args)
    if len(context_out_keys):
        if type(new_context) is tuple:
            # print(new_context)
            new_context = {k: new_context[i] for i, k in enumerate(context_out_keys)}
        elif len(context_out_keys) == 1:
            new_context = {context_out_keys[0]: new_context}
        else:
            raise ValueError('len(context_out_keys) needs to match number of outputs from xf fun')
    # Use the new context for the next step
    if type(new_context) is not dict:
        raise ValueError('xf did not return a context dict: {}'.format(xf))
    context_out = {**context, **new_context}

    return context_out
