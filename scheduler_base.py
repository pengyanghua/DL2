import Queue
import numpy as np
import parameters as pm
from cluster import Cluster
import log

class Scheduler(object):
	def __init__(self, name, trace, logger):
		self.name = name  # e.g., 'DRF'
		self.trace = trace
		if logger is None:
			assert name
			self.logger = log.getLogger(name=name, fh=False)
		else:
			self.logger = logger

		self.cluster = Cluster(self.logger)
		self.curr_ts = 0
		self.end = False

		self.running_jobs = set()
		self.uncompleted_jobs = set()
		self.completed_jobs = set()

		self.data = None  # all state action pairs in one ts
		self.rewards = []

	def step(self):
		# step by one timeslot
		assert not self.end
		self._prepare()
		self._schedule()
		self._progress()
		if len(self.completed_jobs) == pm.TOT_NUM_JOBS:
			self.end = True
		self.curr_ts += 1
		return self.data

	def get_results(self):
		# get final results, including avg jct, makespan and avg reward
		jct_list = [(job.end_time - job.arrv_time + 1.0) for job in self.completed_jobs]
		makespan = max([job.end_time+1.0 for job in self.completed_jobs])
		assert jct_list
		return (len(self.completed_jobs), 1.0*sum(jct_list)/len(jct_list), makespan, sum(self.rewards)/len(self.rewards))

	def get_job_jcts(self):
		jcts = dict()
		for job in self.completed_jobs:
			jcts[job.id] = job.end_time - job.arrv_time + 1.0
		return jcts

	def _prepare(self):
		self.cluster.clear()
		self.data = []
		self.running_jobs.clear()
		if self.curr_ts in self.trace:
			for job in self.trace[self.curr_ts]:
				job.reset() # must reset since it is trained for multiple epochs
				self.uncompleted_jobs.add(job)
				self.logger.debug(job.info())
		for job in self.uncompleted_jobs:
			job.num_workers = 0
			job.curr_worker_placement = []
			if pm.PS_WORKER:
				job.num_ps = 0
				job.curr_ps_placement = []
		# sort based on used resources from smallest to largest for load balancing
		self.node_used_resr_queue = Queue.PriorityQueue()
		for i in range(pm.CLUSTER_NUM_NODES):
			self.node_used_resr_queue.put((0, i))

	def _schedule(self):
		self.logger.info("This method is to be implemented on child class!")

	def _progress(self):
		reward = 0
		for job in self.running_jobs.copy():
			epoch = job.step()
			reward += epoch / job.num_epochs
			if job.progress >= job.real_num_epochs:
				if pm.FINE_GRAIN_JCT:
					job.end_time = self.curr_ts - 1 + job.get_run_time_in_ts()
				else:
					job.end_time = self.curr_ts
				# self.running_jobs.remove(job)
				self.uncompleted_jobs.remove(job)
				self.completed_jobs.add(job)
		if pm.NUM_UNCOMPLETED_JOB_REWARD:
			reward = len(self.uncompleted_jobs)
		self.rewards.append(reward)

	def observe(self):
		'''
		existing resource share of each job: 0-1
		job type 0-8
		job normalized progress 0-1
		num of backlogs: percentage of total number of jobs in the trace
		'''
		# cluster_state = self.cluster.get_cluster_state()
		# for test, first use dominant resource share of each job as input state
		q = Queue.PriorityQueue()
		for job in self.uncompleted_jobs:
			if pm.PS_WORKER:
				if job.num_workers >= pm.MAX_NUM_WORKERS and job.num_ps >= pm.MAX_NUM_WORKERS: # and, not or
					continue
			else:
				if job.num_workers >= pm.MAX_NUM_WORKERS:  # not schedule it any more
					continue
			if pm.JOB_SORT_PRIORITY == "Resource":
				q.put((job.dom_share, job.arrv_time, job))
			elif pm.JOB_SORT_PRIORITY == "Arrival":
				q.put((job.arrv_time, job.arrv_time, job))
			elif pm.JOB_SORT_PRIORITY == "Progress":
				q.put((1-job.progress/job.num_epochs, job.arrv_time, job))

		if pm.ZERO_PADDING:
			state = np.zeros(shape=pm.STATE_DIM)  # zero padding instead of -1
		else:
			state = -1*np.ones(shape=pm.STATE_DIM)
		self.window_jobs = [None for _ in range(pm.SCHED_WINDOW_SIZE)]

		shuffle = np.array([i for i in range(pm.SCHED_WINDOW_SIZE)]) # default keep order
		if pm.JOB_ORDER_SHUFFLE:
			shuffle = np.random.choice(pm.SCHED_WINDOW_SIZE, pm.SCHED_WINDOW_SIZE, replace=False)

		# resource share / job arrival / progress
		for order in shuffle:
			if not q.empty():
				_, _, job = q.get()
				j = 0
				for (input,enable) in pm.INPUTS_GATE: # INPUTS_GATE=[("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
					if enable:
						if input == "TYPE":
							if not pm.INPUT_RESCALE:
								if not pm.TYPE_BINARY:
									state[j][order] = job.type
								else:
									bin_str = "{0:b}".format(job.type).zfill(4)
									for bin_ch in bin_str:
										state[j][order] = int(bin_ch)
										j += 1
									j -= 1
							else:
								state[j][order] = float(job.type)/8
						elif input == "STAY":
							if not pm.INPUT_RESCALE:
								state[j][order] = self.curr_ts - job.arrv_time
							else:
								state[j][order] = float(self.curr_ts - job.arrv_time) / 100
						elif input == "PROGRESS":
							state[j][order] = 1 - job.progress/job.num_epochs
						elif input == "DOM_RESR":
							state[j][order] = job.dom_share
						elif input == "WORKERS":
							if not pm.INPUT_RESCALE:
								state[j][order] = job.num_workers
							else:
								state[j][order] = float(job.num_workers)/pm.MAX_NUM_WORKERS
						elif input == "PS":
							if not pm.INPUT_RESCALE:
								state[j][order] = job.num_ps
							else:
								state[j][order] = float(job.num_ps) / pm.MAX_NUM_WORKERS
						else:
							raise RuntimeError
						j += 1
				self.window_jobs[order] = job

		# backlog = float(max(len(self.uncompleted_jobs) - pm.SCHED_WINDOW_SIZE, 0))/len(pm.TOT_NUM_JOBS)
		self.logger.debug("ts: " + str(self.curr_ts) \
						  + " backlog: " + str(max(len(self.uncompleted_jobs) - pm.SCHED_WINDOW_SIZE, 0)) \
						  + " completed jobs: " + str(len(self.completed_jobs)) \
						  + " uncompleted jobs: " + str(len(self.uncompleted_jobs)))
		return state

	def _state(self, label_job_id, role="worker"): # whether this action selection leads to worker increment or ps increment
		# cluster_state = self.cluster.get_cluster_state()
		input = self.observe()  #  NN input
		label = np.zeros(pm.ACTION_DIM)
		for i in range(pm.SCHED_WINDOW_SIZE):
			job = self.window_jobs[i]
			if job and job.id == label_job_id:
				if pm.PS_WORKER:
					if pm.BUNDLE_ACTION:
						if role == "worker":
							label[i * 3] = 1
						elif role == "ps":
							label[i * 3 + 1] = 1
						elif role == "bundle":
							label[i * 3 + 2] = 1
					else:
						if role == "worker":
							label[i * 2] = 1
						elif role == "ps":
							label[i * 2 + 1] = 1
				else:
					label[i] = 1
		self.data.append((input,label))
