import numpy as np

def majority_voting(masks, voting='hard', weights=None, threshold=0.5):
    """Soft Voting/Majority Rule mask merging; Signature based upon the Scikit-learn VotingClassifier (https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_voting.py#L141)
    
    Parameters
    ----------
    masks : segmentations masks to merge, ndarray
        Expected shape is num_of_masks * 1 * h * w
        Accepts masks in range 0-1 (i.e apply sigmoid before passing to this function)
    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    threshold : for separating between the positive and negative class, default=0.5
        Applied first in case of hard voting and applied last in case of soft voting
    """
    assert len(masks.shape) == 4

    if voting not in ('soft', 'hard'):
        raise ValueError(f"Voting must be 'soft' or 'hard'; got (voting= {voting})")

    for m in masks:
        assert (m >= 0.).all() and (m <= 1.).all()

    if voting == 'hard':
        masks = (masks >= threshold).astype(np.float32)
    
    if weights is None:
        weights = np.array([1] * masks.shape[0])
    else:
        weights = np.array(weights)

    # Broadcasting starts with the trailing (i.e. rightmost) dimensions and works its way left, therefore we move the "mask" dimension to the right
    masks= np.transpose(masks, (1, 2, 3, 0))
    masks = masks * weights
    masks= np.transpose(masks, (3, 0, 1, 2))
    masks = masks.sum(axis=0)

    if voting == 'soft':
        masks = (masks >= (threshold * weights.sum())).astype(np.float32)
    elif voting == 'hard': # Same as doing a majority vote
        masks = (masks > (0.5 * weights.sum())).astype(np.float32)
    
    assert len(masks.shape) == 3
    
    return masks.astype(np.float32)


def test_majority_voting():
    m1 = np.zeros((1,2,2))
    m2 = np.ones((1,2,2))
    m3 = np.array([[[0.4, 0.4],
        [0.4, 0.4]]]) 
    m4 = np.array([[[0.6, 0.6],
        [0.6, 0.6]]]) 
    m5 = np.array([[[0.7, 0.7],
        [0.2, 0.1]]]) 
    m6 = np.array([[[0.55, 0.1],
        [0.2, 0.6]]]) 

    masks = np.stack([m1, m2], axis=0)
    assert (majority_voting(masks, voting='hard') == np.array([[[0., 0.], [0., 0.]]])).all() # since threshold is >

    masks = np.stack([m1, m2], axis=0)
    assert (majority_voting(masks, weights=[2,1]) == np.array([[[0., 0.], [0., 0.]]])).all()

    masks = np.stack([m1, m2, m3], axis=0)
    assert (majority_voting(masks) == np.array([[[0., 0.], [0., 0.]]])).all()

    masks = np.stack([m3, m4], axis=0)
    assert (majority_voting(masks, weights=[2,1]) == np.array([[[0., 0.], [0., 0.]]])).all()
    assert (majority_voting(masks, weights=[1,2]) == np.array([[[1., 1.], [1., 1.]]])).all()

    masks = np.stack([m1, m2, m3, m4, m5], axis=0)
    assert (majority_voting(masks) == np.array([[[1., 1.], [0., 0.]]])).all()

    masks = np.stack([m4, m5, m6], axis=0)
    assert (majority_voting(masks, voting = 'soft') == np.array([[[1., 0.], [0., 0.]]])).all()
    assert (majority_voting(masks, voting = 'soft', weights=[1,2,1]) == np.array([[[1., 1.], [0., 0.]]])).all()
    assert (majority_voting(masks, voting = 'hard') == np.array([[[1., 1.], [0., 1.]]])).all()
    assert (majority_voting(masks, voting = 'hard', weights=[1,2,1]) == np.array([[[1., 1.], [0., 0.]]])).all()

# test_majority_voting()