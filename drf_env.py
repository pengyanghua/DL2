import Queue
import time
import numpy as np
import parameters as pm
from scheduler_base import Scheduler

class DRF_Env(Scheduler):
	# overwrite the scheduling algorithm in Scheduler
	def _schedule(self):
		tic = time.time()
		drf_queue = Queue.PriorityQueue()
		for job in self.uncompleted_jobs:
			drf_queue.put((0, job.arrv_time, job))  # enqueue jobs into drf queue

		while not drf_queue.empty():
			(_, job_arrival, job) = drf_queue.get()
			# bundle one ps and one worker together by default
			_, node = self.node_used_resr_queue.get()
			if pm.PS_WORKER:
				resr_reqs = job.resr_worker + job.resr_ps
			else:
				resr_reqs = job.resr_worker
			succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
			self.node_used_resr_queue.put((np.sum(node_used_resrs), node))
			if succ:
				if pm.PS_WORKER and pm.BUNDLE_ACTION and False:
					self._state(job.id, "bundle")
					job.num_workers += 1
					job.curr_worker_placement.append(node)
					job.num_ps += 1
					job.curr_ps_placement.append(node)
					job.dom_share = np.max(1.0 * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)
				else:
					self._state(job.id, "worker")
					job.num_workers += 1
					job.curr_worker_placement.append(node)
					job.dom_share = np.max(1.0 * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)
					# print "worker", self.data[-1]

					if pm.PS_WORKER:
						self._state(job.id, "ps")
						job.num_ps += 1
						job.curr_ps_placement.append(node)
						job.dom_share = np.max(1.0 * (job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)

				# print "ps", self.data[-1]
				# a = raw_input("next step?")
				self.running_jobs.add(job)
				if job.num_workers < pm.MAX_NUM_WORKERS and job.num_ps < pm.MAX_NUM_WORKERS:
					drf_queue.put((job.dom_share, job_arrival, job))

			else: # fail to alloc resources, try other jobs
				# continue
				break

		toc = time.time()
		self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
		for job in self.uncompleted_jobs:
			self.logger.debug(self.name + ":: scheduling results" +" num_worker: " + str(job.num_workers) +" num_ps: " + str(job.num_ps))


def test():
	import log, trace
	np.random.seed(9973)
	logger = log.getLogger(name="test.log", level="DEBUG")
	job_trace = trace.Trace(logger).get_trace()
	env = DRF_Env("DRF", job_trace, logger)
	while not env.end:
		env.step()
		#print env.observe()
		# print env.data
		# input()
	print env.get_results()
	print env.get_job_jcts()
	for i in range(len(env.trace)):
		if i in env.trace:
			for job in env.trace[i]:
				print i+1, job.id, job.type, job.model






if __name__ == '__main__':
	test()





