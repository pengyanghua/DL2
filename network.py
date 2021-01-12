import numpy as np
import tflearn
import tensorflow as tf
import parameters as pm


class PolicyNetwork:
	def __init__(self, sess, scope, mode, logger):
		self.sess = sess
		self.state_dim = pm.STATE_DIM
		self.action_dim = pm.ACTION_DIM
		self.scope = scope
		self.mode = mode
		self.logger = logger

		self.input, self.output = self._create_nn()
		self.label = tf.placeholder(tf.float32, [None, self.action_dim])
		self.action = tf.placeholder(tf.float32, [None, None])
		self.advantage = tf.placeholder(tf.float32, [None, 1])

		self.entropy = tf.reduce_mean(tf.multiply(self.output, tf.log(self.output + pm.ENTROPY_EPS)))
		self.entropy_weight = pm.ENTROPY_WEIGHT

		if self.mode == "SL":
			if pm.SL_LOSS_FUNCTION == "Mean_Square":
				self.loss = tf.reduce_mean(tflearn.mean_square (self.output, self.label))
			elif pm.SL_LOSS_FUNCTION == "Cross_Entropy":
				self.loss = tf.reduce_mean(tflearn.categorical_crossentropy(self.output,self.label))
			elif pm.SL_LOSS_FUNCTION == "Absolute_Difference":
				self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.output, self.label))
		elif self.mode == "RL":
			self.loss = tf.reduce_mean(tf.multiply(tf.log(tf.reduce_sum(tf.multiply(self.output, self.action), reduction_indices=1, keep_dims=True)), -self.advantage)) \
						+ self.entropy_weight * self.entropy
		#self.loss = tf.reduce_mean(tflearn.mean_square(self.output, self.label))

		self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
		self.gradients = tf.gradients(self.loss, self.weights)

		self.lr = pm.LEARNING_RATE
		if pm.OPTIMIZER == "Adam":
			self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(self.gradients, self.weights))
		elif pm.OPTIMIZER == "RMSProp":
			self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.lr).apply_gradients(zip(self.gradients, self.weights))

		self.weights_phs = []
		for weight in self.weights:
			self.weights_phs.append(tf.placeholder(tf.float32, shape=weight.get_shape()))
		self.set_weights_op = []
		for idx, weights_ph in enumerate(self.weights_phs):
			self.set_weights_op.append(self.weights[idx].assign(weights_ph))

		self.loss_ring_buff = [0 for _ in range(20)]
		self.index_ring_buff = 0


	def _create_nn(self):
		with tf.variable_scope(self.scope):
			# type, arrival, progress, resource
			input = tflearn.input_data(shape=[None, self.state_dim[0], self.state_dim[1]], name="input") # row is info type, column is job

			if pm.JOB_CENTRAL_REPRESENTATION or pm.ATTRIBUTE_CENTRAL_REPRESENTATION:
				if pm.JOB_CENTRAL_REPRESENTATION:
					fc_list = []
					for i in range(self.state_dim[1]):
						if pm.FIRST_LAYER_TANH:
							fc1 = tflearn.fully_connected(input[:, :, i], self.state_dim[0], activation="tanh", name="job_" + str(i))
						else:
							fc1 = tflearn.fully_connected(input[:, :, i], self.state_dim[0], activation="relu", name="job_"+str(i))
						if pm.BATCH_NORMALIZATION:
							fc1 = tflearn.batch_normalization(fc1, name="job_"+str(i)+"_bn")
						fc_list.append(fc1)
				else:
					j = 0
					fc_list = []
					for (key, enable) in pm.INPUTS_GATE:  # INPUTS_GATE=[("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
						if enable:
							if pm.FIRST_LAYER_TANH:
								fc1 = tflearn.fully_connected(input[:, j], pm.SCHED_WINDOW_SIZE, activation="tanh", name=key)
							else:
								fc1 = tflearn.fully_connected(input[:, j], pm.SCHED_WINDOW_SIZE, activation="relu", name=key)
							if pm.BATCH_NORMALIZATION:
								fc1 = tflearn.batch_normalization(fc1, name=key+"_bn")
							fc_list.append(fc1)
							j += 1
				if len(fc_list) == 1:
					merge_net = fc_list[0]
					if pm.BATCH_NORMALIZATION:
						merge_net = tflearn.batch_normalization(merge_net)
				else:
					merge_net = tflearn.merge(fc_list, 'concat', name="merge_net_1")
					if pm.BATCH_NORMALIZATION:
						merge_net = tflearn.batch_normalization(merge_net, name="merge_net_1_bn")
				dense_net_1 = tflearn.fully_connected(merge_net, pm.NUM_NEURONS_PER_FCN, activation='relu', name='dense_net_1')
			else:
				dense_net_1 = tflearn.fully_connected(input, pm.NUM_NEURONS_PER_FCN, activation='relu', name='dense_net_1')
			if pm.BATCH_NORMALIZATION:
				dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_1_bn')

			for i in range(1, pm.NUM_FCN_LAYERS):
				dense_net_1 = tflearn.fully_connected(dense_net_1, pm.NUM_NEURONS_PER_FCN, activation='relu', name='dense_net_' + str(i + 1))
				if pm.BATCH_NORMALIZATION:
					dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_' + str(i + 1) + 'bn')

			if pm.JOB_CENTRAL_REPRESENTATION and pm.NN_SHORTCUT_CONN:  # add shortcut the last layer
				fc2_list = []
				for fc in fc_list:
					merge_net_2 = tflearn.merge([fc, dense_net_1], 'concat')
					if pm.PS_WORKER:
						if pm.BUNDLE_ACTION:
							fc2 = tflearn.fully_connected(merge_net_2, 3, activation='linear')
						else:
							fc2 = tflearn.fully_connected(merge_net_2, 2, activation='linear')
					else:
						fc2 = tflearn.fully_connected(merge_net_2, 1, activation='linear')
					fc2_list.append(fc2)

				if pm.SKIP_TS:
					fc2 = tflearn.fully_connected(dense_net_1, 1, activation='linear')
					fc2_list.append(fc2)
				merge_net_3 = tflearn.merge(fc2_list, 'concat')
				output = tflearn.activation(merge_net_3, activation="softmax", name="policy_output")
			else:
				output = tflearn.fully_connected(dense_net_1, self.action_dim, activation="softmax", name="policy_output")
			return input, output


	def get_sl_loss(self, input, label):
		assert self.mode == "SL"
		return self.sess.run([self.output, self.loss], feed_dict={self.input:input, self.label:label})


	def predict(self, input):
		return self.sess.run(self.output, feed_dict={self.input:input})


	def get_sl_gradients(self, input, label):
		assert self.mode == "SL"
		return self.sess.run([self.entropy, self.loss, self.gradients], feed_dict={self.input:input, self.label:label})


	def get_rl_gradients(self, input, output, action, advantage):
		assert  self.mode == "RL"
		return self.sess.run([self.entropy, self.loss, self.gradients],
							feed_dict={self.input:input, self.output:output, self.action:action,
									   self.advantage:advantage})


	def apply_gradients(self, gradients):
		self.sess.run(self.optimize, feed_dict={i:d for i,d in zip(self.gradients,gradients)})


	def set_weights(self, weights):
		self.sess.run(self.set_weights_op, feed_dict={i:d for i,d in zip(self.weights_phs, weights)})


	def get_weights(self):
		return self.sess.run(self.weights)

	def get_num_weights(self):
		with tf.variable_scope(self.scope):
			total_parameters = 0
			for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope):
				# shape is an array of tf.Dimension
				shape = variable.get_shape()
				# print variable.name
				variable_parameters = 1
				for dim in shape:
					variable_parameters *= dim.value
				# print "varable in each layer {0}".format(variable_parameters)
				total_parameters += variable_parameters
			return total_parameters

	# adjust entropy weight
	def anneal_entropy_weight(self, step):
		if pm.FIX_ENTROPY_WEIGHT:
			self.entropy_weight = pm.ENTROPY_WEIGHT
		else:
			self.entropy_weight = max(pm.MAX_ENTROPY_WEIGHT * 2 / (1 + np.exp(step / pm.ANNEALING_TEMPERATURE)), 0.1)



class ValueNetwork:
	def __init__(self, sess, scope, mode, logger):
		self.sess = sess
		self.state_dim = pm.STATE_DIM
		self.action_dim = pm.ACTION_DIM
		self.scope = scope
		self.mode = mode
		self.logger = logger

		self.input, self.output = self._create_nn()
		self.label = tf.placeholder(tf.float32, [None, self.action_dim])
		self.action = tf.placeholder(tf.float32, [None, None])

		self.entropy_weight = pm.ENTROPY_WEIGHT

		self.td_target = tf.placeholder(tf.float32, [None, 1])
		self.loss = tflearn.mean_square(self.output, self.td_target)

		self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
		self.gradients = tf.gradients(self.loss, self.weights)

		self.lr = pm.LEARNING_RATE
		if pm.OPTIMIZER == "Adam":
			self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(self.gradients, self.weights))
		elif pm.OPTIMIZER == "RMSProp":
			self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.lr).apply_gradients(zip(self.gradients, self.weights))
		self.weights_phs = []
		for weight in self.weights:
			self.weights_phs.append(tf.placeholder(tf.float32, shape=weight.get_shape()))
		self.set_weights_op = []
		for idx, weights_ph in enumerate(self.weights_phs):
			self.set_weights_op.append(self.weights[idx].assign(weights_ph))


	def _create_nn(self):
		with tf.variable_scope(self.scope):
			# type, arrival, progress, resource
			input = tflearn.input_data(shape=[None, self.state_dim[0], self.state_dim[1]], name="input") # row is info type, column is job

			if pm.JOB_CENTRAL_REPRESENTATION or pm.ATTRIBUTE_CENTRAL_REPRESENTATION:
				if pm.JOB_CENTRAL_REPRESENTATION:
					fc_list = []
					for i in range(self.state_dim[1]):
						if pm.FIRST_LAYER_TANH:
							fc1 = tflearn.fully_connected(input[:, :, i], self.state_dim[0], activation="tanh", name="job_" + str(i))
						else:
							fc1 = tflearn.fully_connected(input[:, :, i], self.state_dim[0], activation="relu", name="job_"+str(i))
						if pm.BATCH_NORMALIZATION:
							fc1 = tflearn.batch_normalization(fc1, name="job_"+str(i)+"_bn")
						fc_list.append(fc1)
				else:
					j = 0
					fc_list = []
					for (key, enable) in pm.INPUTS_GATE:  # INPUTS_GATE=[("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
						if enable:
							if pm.FIRST_LAYER_TANH:
								fc1 = tflearn.fully_connected(input[:, j], pm.SCHED_WINDOW_SIZE, activation="tanh", name=key)
							else:
								fc1 = tflearn.fully_connected(input[:, j], pm.SCHED_WINDOW_SIZE, activation="relu", name=key)
							if pm.BATCH_NORMALIZATION:
								fc1 = tflearn.batch_normalization(fc1, name=key+"_bn")
							fc_list.append(fc1)
							j += 1
				if len(fc_list) == 1:
					merge_net = fc_list[0]
					if pm.BATCH_NORMALIZATION:
						merge_net = tflearn.batch_normalization(merge_net)
				else:
					merge_net = tflearn.merge(fc_list, 'concat', name="merge_net_1")
					if pm.BATCH_NORMALIZATION:
						merge_net = tflearn.batch_normalization(merge_net, name="merge_net_1_bn")
				dense_net_1 = tflearn.fully_connected(merge_net, pm.NUM_NEURONS_PER_FCN, activation='relu', name='dense_net_1')
			else:
				dense_net_1 = tflearn.fully_connected(input, pm.NUM_NEURONS_PER_FCN, activation='relu', name='dense_net_1')
			if pm.BATCH_NORMALIZATION:
				dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_1_bn')

			for i in range(1, pm.NUM_FCN_LAYERS):
				dense_net_1 = tflearn.fully_connected(dense_net_1, pm.NUM_NEURONS_PER_FCN, activation='relu', name='dense_net_' + str(i + 1))
				if pm.BATCH_NORMALIZATION:
					dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_' + str(i + 1) + 'bn')

			if pm.JOB_CENTRAL_REPRESENTATION and pm.NN_SHORTCUT_CONN:  # a more layer if critic adds shortcut
				fc2_list = []
				for fc in fc_list:
					merge_net_2 = tflearn.merge([fc, dense_net_1], 'concat')
					if pm.PS_WORKER:
						if pm.BUNDLE_ACTION:
							fc2 = tflearn.fully_connected(merge_net_2, 3, activation='relu')
						else:
							fc2 = tflearn.fully_connected(merge_net_2, 2, activation='relu')
					else:
						fc2 = tflearn.fully_connected(merge_net_2, 1, activation='relu')
					fc2_list.append(fc2)

				if pm.SKIP_TS:
					fc2 = tflearn.fully_connected(dense_net_1, 1, activation='relu')
					fc2_list.append(fc2)
				merge_net_3 = tflearn.merge(fc2_list, 'concat', name='merge_net_3')
				if pm.BATCH_NORMALIZATION:
					merge_net_3 = tflearn.batch_normalization(merge_net_3, name='merge_net_3_bn')
				output = tflearn.fully_connected(merge_net_3, 1, activation="linear", name="value_output")
			else:
				output = tflearn.fully_connected(dense_net_1, 1, activation="linear", name="value_output")
			return input, output

	def get_loss(self, input):
		return self.sess.run(self.loss, feed_dict={self.input: input})


	def predict(self, input):
		return self.sess.run(self.output, feed_dict={self.input:input})


	def get_rl_gradients(self, input, output, td_target):
		return self.sess.run([self.loss, self.gradients],
							feed_dict={self.input:input, self.output:output, self.td_target:td_target})


	def apply_gradients(self, gradients):
		self.sess.run(self.optimize, feed_dict={i:d for i,d in zip(self.gradients,gradients)})


	def set_weights(self, weights):
		self.sess.run(self.set_weights_op, feed_dict={i:d for i,d in zip(self.weights_phs, weights)})


	def get_weights(self):
		return self.sess.run(self.weights)

	def get_num_weights(self):
		with tf.variable_scope(self.scope):
			total_parameters = 0
			for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope):
				# shape is an array of tf.Dimension
				shape = variable.get_shape()
				# print variable.name
				variable_parameters = 1
				for dim in shape:
					variable_parameters *= dim.value
				# print "varable in each layer {0}".format(variable_parameters)
				total_parameters += variable_parameters
			return total_parameters



