import Queue
import time
import numpy as np
import parameters as pm
from scheduler_base import Scheduler


EST_ERROR = 0.0 # change to 0.05 with estimation error


class Optimus_Env(Scheduler):
	# can negatively impact performance when 1. local minimum 2. EST_ERROR make the utility negative,
	# need to use curve fitting for correct implementation of optimus
	def est_util(self, job):
		if job.num_workers == 0:
			return (-np.iinfo(np.int32).max, "worker")
		if pm.PS_WORKER and job.num_ps == 0:
			return (-np.iinfo(np.int32).max, "ps")

		speed = job.step(False) * (1+EST_ERROR*np.random.choice([-1,1],1))
		node_used_resrs, node = self.node_used_resr_queue.get()
		self.node_used_resr_queue.put((np.sum(node_used_resrs), node))

		job.num_workers += 1
		job.curr_worker_placement.append(node)
		speed_2 = job.step(False) * (1+EST_ERROR*np.random.choice([-1,1],1))
		worker_utility = (job.num_epochs - job.progress) / speed - (job.num_epochs - job.progress) / speed_2
		job.num_workers -= 1
		job.curr_worker_placement = job.curr_worker_placement[:-1]

		if pm.PS_WORKER:
			job.num_ps += 1
			job.curr_ps_placement.append(node)
			speed_3 = job.step(False)
			ps_utility = (job.num_epochs - job.progress) / speed - (job.num_epochs - job.progress) / speed_3
			job.num_ps -= 1
			job.curr_ps_placement = job.curr_ps_placement[:-1]
			if ps_utility >= worker_utility:
				return (-ps_utility, "ps")
			else:
				return (-worker_utility, "worker")
		else:
			return (-worker_utility, "worker")

	def _schedule(self):
		tic = time.time()
		opt_queue = Queue.PriorityQueue() # initialize all jobs' utility to be 0
		for job in self.uncompleted_jobs:
			util, role = self.est_util(job)
			opt_queue.put((util, job, role))

		while not opt_queue.empty():
			utility, job, role = opt_queue.get()
			if utility >= 0:
				break
			elif role == "worker" and job.num_workers >= pm.MAX_NUM_WORKERS:
				continue
			elif pm.PS_WORKER and role == "ps" and job.num_ps >= pm.MAX_NUM_WORKERS:
				continue
			else:
				if pm.PS_WORKER and role == "ps":
					resr_reqs = job.resr_ps
				else:
					resr_reqs = job.resr_worker
				_, node = self.node_used_resr_queue.get()
				succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
				self.node_used_resr_queue.put((np.sum(node_used_resrs), node))
				if succ:
					if pm.PS_WORKER and role == "ps":
						self._state(job.id, "ps")
						job.num_ps += 1
						job.curr_ps_placement.append(node)
					else:
						self._state(job.id, "worker")
						job.num_workers += 1
						job.curr_worker_placement.append(node)
					self.running_jobs.add(job)
					util, role = self.est_util(job)
					opt_queue.put((util, job, role))
				else:
					# continue
					break

		toc = time.time()
		self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
		for job in self.uncompleted_jobs:
			self.logger.debug(self.name + ":: scheduling results" + " type: " + str(job.type) + " num_worker: " + str(job.num_workers) +" num_ps: " + str(job.num_ps))



def test():
	import log, trace
	logger = log.getLogger(name="test.log", level="DEBUG")
	job_trace = trace.Trace(logger).get_trace()
	env = Optimus_Env("Optimus", job_trace, logger)
	while not env.end:
		env.step()
	print env.get_results()





if __name__ == '__main__':
	test()




