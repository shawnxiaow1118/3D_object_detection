import numpy as np
import numpy.random as random


def rpn_target(all_anchors, inside_inds,gt_labels, gt_boxes):
	# keep inside anchors
	anchors = all_anchors[inside_inds, :]
	if Debug:
		print('anchors.shape ' + anchors.shape)
		labels = np.empty((len(inside_inds),) dtype=np.float32)
		labels.fill(-1)

		overlaps = bbox_overlaps(
			np.ascontiguousarray(anchors, dtype=np.float32),
			np.ascontiguousarray(gt_boxes, dtype=np.float32))

		# indices of most possible labels for each anchors
		argmax_overlaps = overlaps.argmax(overlaps, axis=1)
		max_overlaps = overlaps[np.arange(len(inside_inds)), argmax_overlaps]

		gt_argmax_overlaps = overlaps.argmax(overlaps, axis=0)
		gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

		gt_argmax_overlaps = np.where(overlaps==gt_max_overlaps)[0]

		# label 0 for background, 1 for object and -1 for nothing
		labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
		# for each gt, the anchor with highest overlap
		labels[gt_argmax_overlaps] = 1
		# 
		labels[max_overlaps > cfg.TRAIN.RPB_POSITIVE_OVERLAP] = 1

		# subsample
		num_pos = int(cfg.TRAIN.RPN_POS_FRACTION*cfg.TRAIN.RPN_BATCH_SIZES)
		pos_inds = np.where(labels == 1)[0]
		if (len(pos_inds) > num_pos):
			disable_inds = random(
				pos_inds, size=(len(pos_inds)-num_pos), replace=False)
			labels[disable_inds] = -1

		num_neg = cfg.TRAIN.RPN_BATCH_SIZES - np.sum(labels == 1)
		neg_inds = np.where(labels == 1)[0]
		if (len(neg_inds) > num_neg):
			disable_inds = random(
				neg_inds, size=(len(neg_inds)-neg_pos), replace=False)
			labels[disable_inds] = -1

		idx_label = np.where(labels != - 1)[0]
		idx_pos = np.where(labels == 1)[0]

		inds = inside_inds[idx_label]
		labels = labels[idx_label]

		pos_inds = inside_inds[idx_pos]
		pos_anchors = anchors[idx_pos]
		pos_gt_boxes = (gt_boxes[argmax_overlaps])[idx_pos]
		targets = bbox(pos_gt_boxes, pos_anchors)

		return inds, pos_inds, labels, targets