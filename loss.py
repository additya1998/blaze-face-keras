import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy

def smooth_l1(x):

	def func1():
		return x**2 * 0.5

	def func2():
		return tf.abs(x) - tf.constant(0.5)

	def f(x): return tf.cond(tf.less(tf.abs(x), tf.constant(1.0)), func1, func2)

	return tf.map_fn(f, x)

@tf.function
def location_loss(gt_boxes, gt_labels, pred_boxes, pred_labels):

	mask = tf.math.greater_equal(gt_labels, 1)
	# gt_boxes_valid = tf.boolean_mask(gt_boxes, valid_tensor)
	# gt_labels_valid = tf.boolean_mask(gt_labels, valid_tensor)
	# pred_boxes_valid = tf.boolean_mask(pred_boxes, valid_tensor)
	# pred_boxes_valid = tf.boolean_mask(pred_boxes, valid_tensor)
	
	gxs = gt_boxes[:, :, 0:1]
	gys = gt_boxes[:, :, 1:2]
	gws = gt_boxes[:, :, 2:3]
	ghs = gt_boxes[:, :, 3:4]

	pxs = pred_boxes[:, :, 0:1]
	pys = pred_boxes[:, :, 1:2]
	pws = pred_boxes[:, :, 2:3]
	phs = pred_boxes[:, :, 3:4]

	lossx = tf.boolean_mask(tf.map_fn(smooth_l1, tf.reshape(gxs - pxs, (-1, 896))), mask)
	lossy = tf.boolean_mask(tf.map_fn(smooth_l1, tf.reshape(gys - pys, (-1, 896))), mask)
	lossw = tf.boolean_mask(tf.map_fn(smooth_l1, tf.reshape(gws - pws, (-1, 896))), mask)
	lossh = tf.boolean_mask(tf.map_fn(smooth_l1, tf.reshape(ghs - phs, (-1, 896))), mask)

	x_sum = tf.reduce_mean(lossx)
	y_sum = tf.reduce_mean(lossy)
	w_sum = tf.reduce_mean(lossw)
	h_sum = tf.reduce_mean(lossh)

	stacked_loss = tf.stack((x_sum, y_sum, w_sum, h_sum))
	location_loss = tf.reduce_mean(stacked_loss)

	return location_loss

def confidence_loss(gt_labels, pred_labels):
	gt_labels_cat = to_categorical(gt_labels)
	loss = categorical_crossentropy(gt_labels_cat, pred_labels)
	return tf.reduce_mean(loss)