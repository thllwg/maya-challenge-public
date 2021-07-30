import numpy as np
from scipy import ndimage


# The following values are in pixels
MIN_OBJ_SIZE = {
    'building': 50,
    'platform': 85,
    'aguada': 500
}

MIN_OBJ_EDGE_ANY = {
    'building': 5, 
    'platform': 7,
    'aguada': 30
}

MIN_OBJ_EDGE_ALL = {
    'building': 7, 
    'platform': 8,
    'aguada': 40
}

MIN_TOTAL = {
    'building': 10, 
    'platform': 10,
    'aguada': 40
}

# MIN_COUNT = {
#     'building': 1, 
#     'platform': 1,
#     'aguada': 1
# }

MIN_OBJ_SIZE_LARGE = {
    'building': 10 , 
    'platform': 15,
    'aguada': 30
}

MIN_OBJ_EDGE_ANY_LARGE  = {
    'building': 15, 
    'platform': 20,
    'aguada': 50
}

MIN_OBJ_EDGE_ALL_LARGE  = {
    'building': 30, 
    'platform': 40,
    'aguada': 60
}

MIN_TOTAL_LARGE  = {
    'building': 50, 
    'platform': 50,
    'aguada': 100
}

# MIN_COUNT_LARGE = {
#     'building': 3, 
#     'platform': 1,
#     'aguada': 1
# }

def label_on_edge(ori_labeled_mask, value, check = 'any'):
    '''
    Check can be either any (i.e any pixel) touching the boundary, or 'all' (i.e. all pixels touching the boundary)
    '''
    labeled_mask = ori_labeled_mask.copy()
    labeled_mask = np.squeeze(labeled_mask)
            
    h = labeled_mask.shape[0] - 1
    wh = np.where(labeled_mask == value)
    if check == 'any':
        return ( 0 in wh[0] or 0 in wh[1] or h in wh[0] or h in wh[1])
    elif check == 'all':
        return np.all(wh[0] == 0) or  np.all(wh[1] == 0)  or np.all(wh[0] == h) or np.all(wh[1] == h)

def test_label_on_edge():
    mask = np.array([[0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,1,1,0],
    [0,0,1,1,0],
    [0,0,0,0,0]])
    assert label_on_edge(mask, 1, check = 'all') == False
    assert label_on_edge(mask, 1, check = 'any') == False

    mask = np.array([[0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,1,1,1],
    [0,0,1,1,0],
    [0,0,0,0,0]])
    assert label_on_edge(mask, 1, check = 'all') == False
    assert label_on_edge(mask, 1, check = 'any') == True
    
    mask = np.array([[0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,0,0]])
    assert label_on_edge(mask, 1, check = 'all') == True
    assert label_on_edge(mask, 1, check = 'any') == True

    mask = np.array([[0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,2,2,0]])
    assert label_on_edge(mask, 1, check = 'all') == False
    assert label_on_edge(mask, 1, check = 'any') == False

    assert label_on_edge(mask, 2, check = 'all') == True
    assert label_on_edge(mask, 2, check = 'any') == True
    print('Tests passed!' )


def mask_cleanup(ori_mask, min_obj_size, min_obj_edge_any, min_obj_edge_all, min_total, min_obj_count = 1):
    '''
    Steps
        1. Verify that total > minimum (return 0s if not the case)
        2. Create polygons/labels
        3. Remove small polygons
        4. Fill holes
    '''
    mask = ori_mask.copy()
    if np.count_nonzero(mask) == 0:
        return mask

    # 2 - connectivity
    # st = np.array([[1,1,1],[1,1,1], [1,1,1]])


    # We don't have a case when the mask takes more than 55% of the space
    # Temporary disabled due to Issue #60
    #assert mask.mean() <= .55

    # On the other hand, if the predictions are less than a minimum than its better not to predict anything
    fh = ndimage.binary_fill_holes(mask)
    if fh.sum() < min_total: 
        mask.fill(0)
        return mask

    # Remove small polygons 
    labels, lc = ndimage.label(mask)#, structure=st)

    if lc < min_obj_count:
        mask.fill(0)
        return mask

    for i in range(1, lc + 1):
        if label_on_edge(labels, i, check = 'all'):
            labels[labels == i] = 0
        elif label_on_edge(labels, i, check = 'any') and np.count_nonzero(labels == i) < min_obj_edge_any:
            labels[labels == i] = 0
        elif np.count_nonzero(labels == i) < min_obj_size:
            labels[labels == i] = 0
        
    #Finally fill holes (note: fill holes also removes the labels)
    mask = ndimage.binary_fill_holes(labels) 

    return mask

# test_label_on_edge()

def test_mask_cleanup():
    m = np.ones((3,3))
    mc = mask_cleanup(m, min_obj_size = 5, min_obj_edge_any = 5, min_obj_edge_all = 5, min_total =10)
    assert np.all(mc == 0) # since 9 < 10 (min_total)

    # # Single objects

    m = np.zeros((10,10))
    m[3:6,3:6] = 1
    mc = mask_cleanup(m, min_obj_size = 5, min_obj_edge_any = 5, min_obj_edge_all = 5, min_total =10)
    assert np.all(mc == 0) # since 9 < 10 (min_total)


    m = np.zeros((10,10))
    m[3:7,3:7] = 1
    mc = mask_cleanup(m, min_obj_size = 5, min_obj_edge_any = 5, min_obj_edge_all = 5, min_total =10)
    assert np.all(mc == m)  # since 16 > 10 (none apply)

    m = np.zeros((10,10))
    m[0:,0] = 1
    mc = mask_cleanup(m, min_obj_size = 2, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =1)
    assert np.all(mc == m) # since 10 > 8 (!min_obj_edge_all, none apply)

    m = np.zeros((10,10))
    m[0:7,0] = 1
    mc = mask_cleanup(m, min_obj_size = 2, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =1)
    assert np.all(mc == 0) # since 7 < 8 (min_obj_edge_all)

    m = np.zeros((10,10))
    m[0:7,1] = 1
    mc = mask_cleanup(m, min_obj_size = 2, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =1)
    assert np.all(mc == m) # since 7 > 5 (!min_obj_edge_any, none apply)

    m = np.zeros((10,10))
    m[0:7,1] = 1
    mc = mask_cleanup(m, min_obj_size = 2, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =1)
    assert np.all(mc == m) # since 7 > 5 (min_obj_edge_any)

    m = np.zeros((10,10))
    m[5,5] = 1
    mc = mask_cleanup(m, min_obj_size = 2, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =1)
    assert np.all(mc == 0) # since 1 < 2 (min_obj_size)

    #Multiple objects

    m = np.zeros((10,10))
    m[2:4,2:4] = 1
    mc = mask_cleanup(m, min_obj_size = 3, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =5)
    assert np.all(mc == 0) # since 4 < 5 (min_total)
    m[6:8,6:8] = 1
    mc = mask_cleanup(m, min_obj_size = 3, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =5)
    assert np.all(mc == m) # since 8 > 5 (!min_total, none apply)
    mc = mask_cleanup(m, min_obj_size = 5, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =5)
    assert np.all(mc == 0) # since 4 < 5 (min_obj_size)

    m1, m2 = np.zeros((10,10)), np.zeros((10,10))
    m1[2:5,2:5] = 1
    m1[6:8,6:8] = 1
    m2[2:5,2:5] = 1
    mc = mask_cleanup(m1, min_obj_size = 5, min_obj_edge_any = 5, min_obj_edge_all = 8, min_total =5)
    assert np.all(mc == m2) # since 4 < 5 (min_obj_size, one case apply)
    
    m1, m2 = np.zeros((10,10)), np.zeros((10,10))
    m1[0:7,9] = 1
    m1[4:8,4:8] = 1
    m2[4:8,4:8] = 1
    mc = mask_cleanup(m1, min_obj_size = 1, min_obj_edge_any = 2, min_obj_edge_all = 8, min_total =5)
    assert np.all(mc == m2) # since 7 < 8 (min_obj_edge_all, one case apply)

    mc = mask_cleanup(m1, min_obj_size = 1, min_obj_edge_any = 2, min_obj_edge_all = 6, min_total =5)
    assert np.all(mc == m1) # since 7 > 6 (!min_obj_edge_all, none apply)
    

    print('All assertions passed!')
