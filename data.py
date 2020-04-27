import numpy as np 
import tensorflow as tf
from utils import read_tfRecord, preprocess_input
from utils import center_form_to_corner_form, corner_form_to_center_form, convert_boxes_to_locations
from utils import generate_priors, assign_priors


class UplaraDataset:

	def __init__(self, fpath='/home/additya/uplara/blaze-face/sample_train.tfRecord', \
						batch_size=10, \
						train=True, \
						prior_specs = [((16, 0.2, [1, 0.5])), (8, 0.55, [1, 0.5, 0.33, 2, 3, 0.2])]):
		
		self.fpath = fpath
		self.batch_size = batch_size
		self.train = train
		self.prior_specs = prior_specs

		self.center_form_priors = tf.convert_to_tensor(generate_priors(self.prior_specs))
		self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)


	def preprocess_fn(self, image, labels, boxes, probability_tensor):
		return tf.py_function(preprocess_input, \
							  [image, labels, boxes, probability_tensor], \
							  (image.dtype, labels.dtype, boxes.dtype)
		)


	def assign_priors_fn(self, image, labels, boxes):
		return tf.py_function(assign_priors, \
							  [image, labels, boxes, self.corner_form_priors, 0.5], \
							  (image.dtype, labels.dtype, boxes.dtype)
		)


	def get_data(self):

		dataset = tf.data.TFRecordDataset(self.fpath)
		dataset = dataset.map(lambda x:read_tfRecord(x))
		dataset = dataset.map(self.preprocess_fn)
		dataset = dataset.map(self.assign_priors_fn)
		if self.train:
			dataset = dataset.shuffle(True)
		dataset = dataset.batch(self.batch_size)

		return dataset

