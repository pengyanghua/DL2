"""Simple example on how to log scalars and images to tensorboard without tensor ops.
License: Copyleft
"""
__author__ = "Michael Gygli"

import tensorflow as tf
from StringIO import StringIO
import matplotlib.pyplot as plt
import numpy as np


class Logger(object):
	"""Logging in tensorboard without tensorflow ops."""

	def __init__(self, log_dir):
		"""Creates a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)

	def add_graph(self, graph):
		self.writer.add_graph(graph)

	def add_scalar(self, tag, value, step):
		"""Log a scalar variable.
		Parameter
		----------
		tag : basestring
			Name of the scalar
		value
		step : int
			training iteration
		"""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
													 simple_value=value)])
		self.writer.add_summary(summary, step)

	def add_text(self, tag, value, step):
		text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
		meta = tf.SummaryMetadata()
		meta.plugin_data.plugin_name = "text"
		summary = tf.Summary()
		summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
		self.writer.add_summary(summary, step)

	def add_images(self, tag, images, step):
		"""Logs a list of images."""

		im_summaries = []
		for nr, img in enumerate(images):
			# Write the image to a string
			s = StringIO()
			plt.imsave(s, img, format='png')

			# Create an Image object
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
									   height=img.shape[0],
									   width=img.shape[1])
			# Create a Summary value
			im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
												 image=img_sum))

		# Create and write Summary
		summary = tf.Summary(value=im_summaries)
		self.writer.add_summary(summary, step)

	def add_histogram(self, tag, value, step, bins=1000):
		"""Logs the histogram of a list/vector of values."""
		# Convert to a numpy array
		value = np.array(value)

		# Create histogram using numpy
		counts, bin_edges = np.histogram(value, bins=bins)

		# Fill fields of histogram proto
		hist = tf.HistogramProto()
		hist.min = float(np.min(value))
		hist.max = float(np.max(value))
		hist.num = int(np.prod(value.shape))
		hist.sum = float(np.sum(value))
		hist.sum_squares = float(np.sum(value ** 2))

		# Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
		# See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
		# Thus, we drop the start of the first bin
		bin_edges = bin_edges[1:]

		# Add bin edges and counts
		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)

		# Create and write Summary
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)

	def flush(self):
		self.writer.flush()