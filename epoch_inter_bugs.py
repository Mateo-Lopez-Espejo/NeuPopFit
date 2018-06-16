import numpy as np

# PreStimSilence
a = np.array([[502.11, 502.21],
              [502.41, 502.51]])
#
b = np.array([[502.11, 502.41],
              [502.41, 502.71]])

def epoch_intersection(a, b):
    '''
    Compute the intersection of the epochs. Only regions in a which overlap with
    b will be kept.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    intersection : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the difference of a and b.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result:  [  ]       [ ]             []
    '''
    # Convert to a list and then sort in reversed order such that pop() walks
    # through the occurences from earliest in time to latest in time.
    a = a.tolist()
    a.sort(reverse=True)
    b = b.tolist()
    b.sort(reverse=True)

    intersection = []
    if len(a)==0 or len(b)==0:
        # lists are empty, just exit
        result = np.array([])
        return result

    lb, ub = a.pop()
    lb_b, ub_b = b.pop()

    while True:
        if lb > ub_b:
            #           [ a ]
            #     [ b ]
            # Current epoch in b ends before current epoch in a. Move onto
            # the next epoch in b.
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif ub <= lb_b:
            #   [  a    ]
            #               [ b        ]
            # Current epoch in a ends before current epoch in b. Add bounds
            # and move onto next epoch in a.
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb == lb_b) and (ub == ub_b):
            #   [  a    ]
            #   [  b    ]
            # Current epoch in a matches epoch in b.
            try:
                intersection.append((lb, ub))
                lb, ub = a.pop()
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= ub_b):
            #   [  a    ]
            #     [ b ]
            # Current epoch in b is fully contained in the  current epoch
            # from a. Save everything in
            # a up to the beginning of the current epoch of b. However, keep
            # the portion of the current epoch in a
            # that follows the end of the current epoch in b so we can
            # detremine whether there are additional epochs in b that need
            # to be cut out..
            intersection.append((lb_b, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= lb_b) and (ub <= ub_b):
            #   [  a    ]
            #     [ b        ]
            # Current epoch in b begins in a, but extends past a.
            intersection.append((lb_b, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (lb <= ub_b):
            #   [  a    ]
            # [       b     ]
            # Current epoch in a is fully contained in b
            intersection.append((lb, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (ub < ub_b) and (lb > ub_b):
            #   [  a    ]
            # [ b    ]
            intersection.append((lb, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        else:
            # This should never happen.
            m = 'Unhandled epoch boundary condition. Contact the developers.'
            raise SystemError(m)

    # Add all remaining epochs from a
    #intersection.extend(a[::-1])
    result = np.array(intersection)
    if result.size == 0:
        raise Warning("Epochs did not intersect, resulting array"
                    " is empty.")
    return result

# both elements in a are contained in b, the intersection thus should be a, which is not the case
c = epoch_intersection(a, b)
print('a:\n\n{}\n\nb:\n\n{}\n\nintersection:\n\n{}'.format(a, b, c))
