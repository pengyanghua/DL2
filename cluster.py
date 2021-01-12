import numpy as np
import parameters as pm


class Cluster:
	def __init__(self, logger):
		# 0 means available
		self.logger = logger

		self.CLUSTER_RESR_CAPS = np.array([pm.CLUSTER_NUM_NODES * pm.NUM_RESR_SLOTS for i in range(pm.NUM_RESR_TYPES)])
		self.NODE_RESR_CAPS = np.array([pm.NUM_RESR_SLOTS for i in range(pm.NUM_RESR_TYPES)])
		self.cluster_state = np.zeros(shape=(pm.NUM_RESR_TYPES, pm.CLUSTER_NUM_NODES*pm.NUM_RESR_SLOTS))
		self.nodes_used_resrs = np.zeros(shape=(pm.CLUSTER_NUM_NODES, pm.NUM_RESR_TYPES))


	def alloc(self, resr_reqs, node):
		# allocate resources for one task on a node
		if np.any(np.greater(self.nodes_used_resrs[node] + resr_reqs, self.NODE_RESR_CAPS)):  # resource not enough
			return False,self.nodes_used_resrs[node]
		else:
			self.nodes_used_resrs[node] += resr_reqs
			for i in range(pm.NUM_RESR_TYPES):
				resr_req = resr_reqs[i]
				if resr_req > 0:
					start_index = node*pm.NUM_RESR_SLOTS
					for j in range(pm.NUM_RESR_SLOTS):
						if self.cluster_state[i, j+start_index] == 0:
							self.cluster_state[i, j+start_index] = 1  # instead of job.id/pm.TOT_NUM_JOBS
							resr_req -= 1
							if resr_req == 0:
								break
			return True,self.nodes_used_resrs[node]


	def get_cluster_state(self):
		return self.cluster_state.copy()

	def get_cluster_util(self):
		utils = []
		for i in range(pm.NUM_RESR_TYPES):
			util = float(np.sum(self.nodes_used_resrs[:,i])) / self.CLUSTER_RESR_CAPS[i]
			utils.append(util)

		return utils

	def clear(self):
		self.cluster_state.fill(0)
		self.nodes_used_resrs.fill(0)

