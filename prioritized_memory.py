import numpy as np
import collections
import parameters as pm


class SumTree(object):
	"""
	This SumTree code is modified version and the original code is from:
	https://github.com/jaara/AI-blog/blob/master/SumTree.py

	Story the data with it priority in tree and data frameworks.
	"""

	def __init__(self, capacity):
		self.capacity = capacity  # for all priority values
		self.tree = np.zeros(2 * capacity - 1)
		# [--------------Parent nodes-------------][-------leaves to recode priority-------]
		#             size: capacity - 1                       size: capacity
		self.data = np.zeros(capacity, dtype=object)  # for all transitions
		# [--------------data frame-------------]
		#             size: capacity
		# self.Transition = collections.namedtuple('Transition', ('state_outer', 'output_outer', 'action_outer', 'state_inner', 'output_inner', 'action_inner', 'reward'))
		self.full = False
		self.data_pointer = 0
		self.evict_pq = np.zeros(capacity) # a new problem is introduced by this policy

		self.counter = 0

	def add(self, p, data):
		self.evict_pq[self.data_pointer] = p

		tree_idx = self.data_pointer + self.capacity - 1
		self.data[self.data_pointer] = data  # update data_frame
		self.update(tree_idx, p)  # update tree_frame

		self.data_pointer += 1
		if self.data_pointer >= self.capacity:  # replace when exceed the capacity
			self.data_pointer = 0
			self.full = True
		if pm.PRIORITY_MEMORY_EVICT_PRIORITY:
			if self.full:
				self.data_pointer = np.argmin(self.evict_pq)
				self.counter += 1
				# if self.counter % 1000 == 0:
				# 	print [_.reward for _ in self.data]
				# 	print self.evict_pq
				# 	print min(self.evict_pq), max(self.evict_pq)


	def update(self, tree_idx, p):
		self.evict_pq[tree_idx - self.capacity + 1] = p

		change = p - self.tree[tree_idx]
		self.tree[tree_idx] = p
		# then propagate the change through tree
		while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
			tree_idx = (tree_idx - 1) / 2 # result is forced to be int
			self.tree[tree_idx] += change


	def get_leaf(self, v):
		"""
		Tree structure and array storage:

		Tree index:
			 0         -> storing priority sum
			/ \
		  1     2
		 / \   / \
		3   4 5   6    -> storing priority for transitions

		Array type for storing:
		[0,1,2,3,4,5,6]
		"""
		parent_idx = 0
		while True:     # the while loop is faster than the method in the reference code
			cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
			cr_idx = cl_idx + 1
			if cl_idx >= len(self.tree):        # reach bottom, end search
				leaf_idx = parent_idx
				break
			else:       # downward search, always search for a higher priority node
				if v <= self.tree[cl_idx]:
					parent_idx = cl_idx
				else:
					v -= self.tree[cl_idx]
					parent_idx = cr_idx

		data_idx = leaf_idx - self.capacity + 1
		return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

	@property
	def total_p(self):
		return self.tree[0]  # the root

	def list_leaves(self):
		for parent_idx in range(0, self.capacity - 1):
			print "parent idx", parent_idx, "value", self.tree[parent_idx]

		for tree_idx in range(self.capacity - 1, 2 * self.capacity - 1):
			print "tree idx", tree_idx, "value", self.tree[tree_idx]


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
	"""
	This SumTree code is modified version and the original code is from:
	https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
	"""
	epsilon = 0.01  # small amount to avoid zero priority
	alpha = 0.9  # [0~1] convert the importance of TD error to priority
	beta = 0.1  # importance-sampling, from initial value increasing to 1
	beta_increment_per_sampling = 0.0001
	if pm.PRIORITY_MEMORY_SORT_REWARD:
		abs_err_upper = 7.  # reward
		if pm.MEAN_REWARD_BASELINE:
			abs_err_upper = 4.
	else:
		abs_err_upper = 1.  # clipped abs error

	def __init__(self, maxlen):
		self.tree = SumTree(maxlen)
		self.Transition = collections.namedtuple('Transition', ('state', 'output', 'action', 'reward'))
		self.sample_rewards = collections.deque(maxlen=maxlen) # smaller one like maxline/10 may lead to bias
		self.store_rewards = collections.deque(maxlen=maxlen)

	def store(self, state, output, action, reward):
		transition = self.Transition(state, output, action, reward)
		self.store_rewards.append(reward)

		if pm.PRIORITY_MEMORY_SORT_REWARD and pm.MEAN_REWARD_BASELINE:
			p = max(1, reward - sum(self.store_rewards)/len(self.store_rewards)) # p can not be assigned 0 due to ISWeights.append(np.power(prob/min_prob, -self.beta))
		else:
			p = np.max(self.tree.tree[-self.tree.capacity:])   # 1.5 may be too large
			if p == 0:
				p = self.abs_err_upper/1.2
		self.tree.add(p, transition)   # set the max p for new p

	def sample(self, n):
		b_idx = []
		b_memory = []
		ISWeights = []
		pri_seg = self.tree.total_p / n       # priority segment
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # max = 1

		min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
		for i in range(n):
			a, b = pri_seg * i, pri_seg * (i + 1)
			# a, b = pri_seg * i, min(pri_seg * (i + 3), pri_seg*n) # introduce bias, do not consider any more
			v = np.random.uniform(a, b)
			idx, p, data = self.tree.get_leaf(v)
			prob = p / self.tree.total_p
			# ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
			ISWeights.append(np.power(prob/min_prob, -self.beta)) # higher prob, higher beta -> lower weights
			b_idx.append(idx)
			# b_idx[i] = idx
			# print i, "here"
			# print b_memory[i], type(data)
			# b_memory[i, :] = data
			b_memory.append(data)
			self.sample_rewards.append(data.reward)
		return b_idx, b_memory, ISWeights

	def update(self, tree_idx, abs_errors):
		abs_errors += self.epsilon  # convert to abs and avoid 0
		clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
		ps = np.power(clipped_errors, self.alpha)
		for ti, p in zip(tree_idx, ps):
			self.tree.update(ti, p)

	def avg_reward(self): # just for compatibility
		# assert len(self.sample_rewards) > 0
		# return sum(self.sample_rewards)/len(self.sample_rewards)
		assert  len(self.store_rewards) > 0
		return sum(self.store_rewards) / len(self.store_rewards)

	def full(self):
		return self.tree.full