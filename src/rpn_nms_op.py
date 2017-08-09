def filter_boxes(boxes, min_size):
    '''Remove all boxes with any side smaller than min_size.'''
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep



def rpn_nms_generator(
    stride, img_width, img_height, img_scale=1,
    nms_thresh   =CFG.TRAIN.RPN_NMS_THRESHOLD,
    min_size     =CFG.TRAIN.RPN_NMS_MIN_SIZE,
    nms_pre_topn =CFG.TRAIN.RPN_NMS_PRE_TOPN,
    nms_post_topn=CFG.TRAIN.RPN_NMS_POST_TOPN):


    def rpn_nms(scores, deltas, anchors, inside_inds):
        # 1. Generate proposals from box deltas and shifted anchors
        #batch_size, H, W, C = scores.shape
        #assert(C==2)
        scores = scores.reshape((-1, 2,1))
        scores = scores[:,1,:]
        deltas = deltas.reshape((-1, 4))

        scores = scores[inside_inds]
        deltas = deltas[inside_inds]
        anchors = anchors[inside_inds]

        # Convert anchors into proposals via box transformations
        proposals = box_transform_inv(anchors, deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, img_width, img_height)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep      = filter_boxes(proposals, min_size*img_scale)
        proposals = proposals[keep, :]
        scores    = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if nms_pre_topn > 0:
            order = order[:nms_pre_topn]
            proposals = proposals[order, :]
            scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if nms_post_topn > 0:
            keep = keep[:nms_post_topn]
            proposals = proposals[keep, :]
            scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        roi_scores=scores.squeeze()

        num_proposals = len(proposals)
        batch_inds = np.zeros((num_proposals, 1), dtype=np.float32)
        rois = np.hstack((batch_inds, proposals))

        return rois, roi_scores
    return rpn_nms



def tf_rpn_nms(
    scores, deltas, anchors, inside_inds,
    stride, img_width,img_height,img_scale,
    nms_thresh, min_size, nms_pre_topn, nms_post_topn,
    name='rpn_mns'):

    #<todo>
    #assert batch_size == 1, 'Only single image batches are supported'

    rpn_nms = rpn_nms_generator(stride, img_width, img_height, img_scale, nms_thresh, min_size, nms_pre_topn, nms_post_topn)
    return  \
        tf.py_func(
            rpn_nms,
            [scores, deltas, anchors, inside_inds],
            [tf.float32, tf.float32],
        name = name)