def jump(x, t=0, state=0, cube=None, law=uniform, wait=10, *vargs, seed=None, **options):
    x = np.array(x, dtype='float64')   # caution: must specify the datatype, in-place operation below
    d = x.size
    
    if state is None:
        state = 0
        
    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    
    if not incube(x, cube):
        raise Exception(MESSAGE)
        
    if seed is not None and t==0:
        np.random.seed(seed)
        
    y = law(0, **extract_options(options, 'law'))
    if state == 0:
        if cube[0, 0] < x[0] + y < cube[0, 1]:
            x[0] += y
        state += 1
    else:
        window = int(np.random.exponential(wait))
        stop = min(d - 1, state + window)
        if incube(x[state:stop+1] + y, cube[state:stop+1, :]):
            x[state:stop+1] += y
        state = stop + 1 if stop + 1 < d else 0
    return x, state


def jump2(x, t=0, state=0, cube=None, law=uniform, wait=5, *vargs, seed=None, **options):
    x = np.array(x, dtype='float64')   # caution: must specify the datatype, in-place operation below
    d = x.size
    
    if state is None:
        state = 0
        
    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    
    if not incube(x, cube):
        raise Exception(MESSAGE)
        
    if seed is not None and t==0:
        np.random.seed(seed)
        
    y = law(0, **extract_options(options, 'law'))
    if state == 0:
        if cube[0, 0] < x[0] + y < cube[0, 1]:
            x[0] += y
        state += 1
    else:
        window = int(np.random.exponential(wait))
        stop = min(d - 1, state + window)
        if incube(x[state] + y, cube[state:state+1, :]):
            x[state:stop+1] = x[state] + y
        state = stop + 1 if stop + 1 < d else 0
    return x, state


def jump3(x, t=0, state=0, cube=None, law=uniform, wait=5, *vargs, seed=None, **options):
    x = np.array(x, dtype='float64')   # caution: must specify the datatype, in-place operation below
    d = x.size
    
    if state is None:
        state = 0
        
    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    
    if not incube(x, cube):
        raise Exception(MESSAGE)
        
    if seed is not None and t==0:
        np.random.seed(seed)
        
    y = law(0, **extract_options(options, 'law'))
    if state == 0:
        if cube[0, 0] < x[0] + y < cube[0, 1]:
            x[0] += y
        state += 1
    else:
        window = int(np.random.exponential(wait))
        stop = min(d - 1, state + window)
        if incube(x[d-state] + y, cube[d-state:d-state+1, :]):
            x[d-stop:d-state+1] = x[d-state] + y
        state = stop + 1 if stop + 1 < d else 0
    return x, state
