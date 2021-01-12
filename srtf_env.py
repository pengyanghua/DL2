import Queue
import time
import numpy as np
import parameters as pm
from scheduler_base import Scheduler

class SRTF_Env(Scheduler):
	def _schedule(self):
		tic = time.time()
		srtf_queue = Queue.PriorityQueue()
		for job in self.uncompleted_jobs:
			srtf_queue.put((1-job.progress/job.num_epochs, job.arrv_time, job))  # enqueue jobs into srtf queue

		flag = False
		while not srtf_queue.empty():
			(_, job_arrival, job) = srtf_queue.get()
			# allocate maximal number of workers
			# bundle one ps and one worker together by default
			for i in range(pm.MAX_NUM_WORKERS):
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
						job.dom_share = np.max(1.0 * (
						job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)
					else:
						self._state(job.id, "worker")
						job.num_workers += 1
						job.curr_worker_placement.append(node)
						job.dom_share = np.max(1.0 * (
						job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)

						if pm.PS_WORKER:
							self._state(job.id, "ps")
							job.num_ps += 1
							job.curr_ps_placement.append(node)
							job.dom_share = np.max(1.0 * (
							job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)

					self.running_jobs.add(job)
				else: # fail to alloc resources
					flag = True
					break
			if flag:
				break

		toc = time.time()
		self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
		for job in self.uncompleted_jobs:
			self.logger.debug(self.name + ":: scheduling results" +" num_worker: " + str(job.num_workers))


def test():
	import log, trace
	logger = log.getLogger(name="test.log", level="INFO")
	job_trace = trace.Trace(logger).get_trace()
	env = SRTF_Env("SRTF", job_trace, logger)
	while not env.end:
		env.step()
	print env.get_results()





if __name__ == '__main__':
	test()





