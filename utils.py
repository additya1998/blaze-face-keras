import numpy as np 
import tensorflow as tf

# read data
def read_tfRecord(serialized_example):

	feature_dict = {
		'fname': tf.io.FixedLenFeature([], tf.string),
		'image': tf.io.FixedLenFeature([], tf.string),
		'l_probability': tf.io.FixedLenFeature([], tf.int64),
		'r_probability': tf.io.FixedLenFeature([], tf.int64),
		'min_lx': tf.io.FixedLenFeature([], tf.float32),
		'min_ly': tf.io.FixedLenFeature([], tf.float32),
		'max_lx': tf.io.FixedLenFeature([], tf.float32),
		'max_ly': tf.io.FixedLenFeature([], tf.float32),
		'min_rx': tf.io.FixedLenFeature([], tf.float32),
		'min_ry': tf.io.FixedLenFeature([], tf.float32),
		'max_rx': tf.io.FixedLenFeature([], tf.float32),
		'max_ry': tf.io.FixedLenFeature([], tf.float32)
	}
	example = tf.io.parse_single_example(serialized_example, feature_dict)
	
	image_fpath = example['fname']

	image = tf.image.decode_jpeg(example['image'], channels=3)
	image = tf.cast(image, tf.float32) / 255.0
	image = tf.reshape(image, [512, 512, 3])

	l_probability = tf.cast(example['l_probability'], tf.int32)
	r_probability = tf.cast(example['r_probability'], tf.int32)
	probability_tensor = [l_probability, r_probability]
	
	min_lx = (tf.cast(example['min_lx'], tf.float32) + 0.5)
	min_ly = (tf.cast(example['min_ly'], tf.float32) + 0.5)
	max_lx = (tf.cast(example['max_lx'], tf.float32) + 0.5)
	max_ly = (tf.cast(example['max_ly'], tf.float32) + 0.5)
	min_rx = (tf.cast(example['min_rx'], tf.float32) + 0.5)
	min_ry = (tf.cast(example['min_ry'], tf.float32) + 0.5)
	max_rx = (tf.cast(example['max_rx'], tf.float32) + 0.5)
	max_ry = (tf.cast(example['max_ry'], tf.float32) + 0.5)
	
	min_ly = 1 - min_ly
	max_ly = 1 - max_ly
	min_ry = 1 - min_ry
	max_ry = 1 - max_ry

	(min_ly, max_ly) = (max_ly, min_ly)
	(min_ry, max_ry) = (max_ry, min_ry)
	
	left_coords  = [min_lx, min_ly, max_lx, max_ly]
	right_coords = [min_rx, min_ry, max_rx, max_ry]
	
	labels, boxes = [], []

	labels.append(1); boxes.append(left_coords)
	labels.append(2); boxes.append(right_coords)

	return image, labels, boxes, probability_tensor


# remove invalid data - if foot is not present or bounding box size is 0 
def preprocess_input(image, labels, boxes, probability_tensor):

	new_image = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.BILINEAR) 	
	new_labels, new_boxes = [], []
	for idx in range(probability_tensor.shape[0]):
		if probability_tensor[idx].numpy() != 0:
			box = tf.gather(boxes, idx)
			if box[2] > box[0] and box[3] > box[1]:
				new_labels.append(labels[idx].numpy())
				new_boxes.append([box[0], box[1], box[2], box[3]]) 
	new_labels, new_boxes = tf.convert_to_tensor(new_labels), tf.convert_to_tensor(new_boxes)

	return new_image, new_labels, new_boxes


# [x1, y1, w, h] to [x1, y1, x2, y2]
def center_form_to_corner_form(locations):
	return tf.concat((locations[..., :2] - (locations[..., 2:] / 2),
					 locations[..., :2] + (locations[..., 2:] / 2)), axis=1)


# [x1, y1, x2, y2] to [x1, y1, w, h]
def corner_form_to_center_form(locations):
	return tf.concat(((locations[..., :2] + locations[..., 2:]) / 2,
					  locations[..., 2:] - locations[..., :2]), axis=1)


# preprocess for loss function: check loss function
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance=1, size_variance=1):

	ret = tf.concat([
		(center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
		tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
	], axis=1)

	return ret;


def area_of(left_top, right_bottom):
	res = right_bottom - left_top
	zeros = tf.zeros(res.shape)
	res = tf.where(tf.less_equal(res, 0), zeros, res)
	return res[..., 0] * res[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):

	overlap_left_top = tf.math.maximum(boxes0[..., :2], boxes1[..., :2])
	overlap_right_bottom = tf.math.minimum(boxes0[..., 2:], boxes1[..., 2:])

	overlap_area = area_of(overlap_left_top, overlap_right_bottom)
	area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
	area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
	return overlap_area / (area0 + area1 - overlap_area + eps)


# generate anchor boxes
def generate_priors(prior_specs):
	priors = []
	for (feature_map_size, scale, ratio_list) in prior_specs:
		for x in range(feature_map_size):
			for y in range(feature_map_size):
				x_center, y_center = (x + 0.5) / feature_map_size, (y + 0.5) / feature_map_size
				for ratio in ratio_list:
					w, h = scale * ratio, scale / ratio
					priors.append([x_center, y_center, w, h])
	return priors


# assign input bounding boxes to priors
def assign_priors(image, labels, boxes, corner_form_priors, iou_threshold=0.5):

	# num_priors x num_targets
	ious = iou_of(tf.expand_dims(boxes, 0), tf.expand_dims(corner_form_priors, 1))

	best_target_per_prior = tf.math.reduce_max(ious, axis=1)
	best_target_per_prior_index = tf.math.argmax(ious, axis=1)

	best_prior_per_target = tf.math.reduce_max(ious, axis=0)
	best_prior_per_target_index = tf.math.argmax(ious, axis=0)

	best_target_per_prior_index_list = [x.numpy() for x in best_target_per_prior_index]
	for target_index, prior_index in enumerate(best_prior_per_target_index):
		best_target_per_prior_index_list[prior_index] = target_index

	best_target_per_prior_list = [x.numpy() for x in best_target_per_prior]
	for x in best_prior_per_target_index:
		best_target_per_prior_list[x] = 2.0;

	new_best_target_per_prior_index = tf.convert_to_tensor(best_target_per_prior_index_list)
	new_best_target_per_prior = tf.convert_to_tensor(best_target_per_prior_list)

	final_boxes, final_labels = [], []
	for idx in range(new_best_target_per_prior.shape[0]):
		final_boxes.append(tf.gather(boxes, new_best_target_per_prior_index[idx].numpy()))
		if new_best_target_per_prior[idx].numpy() < iou_threshold:
			final_labels.append(0) # background
		else:
			final_labels.append(labels[new_best_target_per_prior_index[idx]])

	(final_labels, final_boxes) = (tf.convert_to_tensor(final_labels), tf.convert_to_tensor(final_boxes))

	center_form_final_boxes = corner_form_to_center_form(final_boxes)
	center_form_priors = corner_form_to_center_form(corner_form_priors)
	location_form_final_boxes = convert_boxes_to_locations(center_form_final_boxes, center_form_priors)

	return image, final_labels, location_form_final_boxes
