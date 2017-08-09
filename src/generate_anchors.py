import numpy as np 


def generate_base_anchors(base_size=15, 
	ratios=[0.5,1,2], scales=2**np.arange(3, 6)):
	""" Generate base anchors given base size, ratio and scale
	"""
	base_anchor = np.array([1,1,base_size,base_size])-1
	ratio_anchors = generate_ratios(base_anchor, ratios)

	anchors = np.vstack([generate_scales(ratio_anchors[i,:], scales)
		for i in xrange(ratio_anchors.shape[0])])

	return anchors



def anchor_property(anchor):
	""" return center, width, height
	"""
	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	x_ctr = anchor[0] + 0.5*(w-1)
	y_ctr = anchor[1] + 0.5*(h-1)

	return w, h, x_ctr, y_ctr


def make_anchors(ws, hs, x_ctr, y_ctr):
	ws = ws[:, np.newaxis]
	hs = hs[:, np.newaxis]
	anchors = np.hstack((x_ctr - 0.5*(ws-1),
						 y_ctr - 0.5*(hs-1),
						 x_ctr + 0.5*(ws-1),
						 y_ctr + 0.5*(hs-1)))
	return anchors


def generate_ratios(anchor, ratios):
	w,h,x_ctr,y_ctr = anchor_property(anchor)
	size = w*h
	size_ratios = size/ratios
	ws = np.round(np.sqrt(size_ratios))
	hs = np.round(ws*ratios)
	anchors = make_anchors(ws, hs, x_ctr, y_ctr)
	return anchors

def generate_scales(anchor, scales):
	w,h,x_ctr,y_ctr = anchor_property(anchor)
	ws = w*scales
	hs = h*scales
	anchors = make_anchors(ws, hs, x_ctr, y_ctr)
	return anchors


## testing
# anchor = np.array([0,0,15,15])

# roperty = anchor_property(anchor)
# print(roperty)

# anchors = generate_ratios(anchor, [0.5,1,2])
# print(anchors)

# anchors = generate_scales(anchor, 2**np.arange(3, 6))
# print(anchors)

anchors = generate_base_anchors()
print(anchors)