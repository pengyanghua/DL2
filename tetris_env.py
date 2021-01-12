import Queue
import time
import numpy as np
import parameters as pm
from scheduler_base import Scheduler

class Tetris_Env(Scheduler):

	def _schedule(self):
		tic = time.time()
		if len(self.uncompleted_jobs) > 0:
			node_used_resr_queue = Queue.PriorityQueue()
			for i in range(pm.CLUSTER_NUM_NODES):
				node_used_resr_queue.put((i, np.zeros(pm.NUM_RESR_TYPES)))  # this queue is sorted based on node id instead of available resources

			while not node_used_resr_queue.empty():
				node, used_resrs = node_used_resr_queue.get()
				# calculate score
				mean_resr_score = dict()
				mean_align_score = dict()
				for job in self.uncompleted_jobs:
					if pm.PS_WORKER:
						resr = job.resr_ps + job.resr_worker
					else:
						resr = job.resr_worker
					mean_resr_score[job] = np.sum(resr) * (1 - job.progress / job.num_epochs)
					mean_align_score[job] = np.sum((pm.NUM_RESR_SLOTS - used_resrs) * resr)
				weight = (float(sum(mean_align_score.values())) / len(mean_align_score)) / (float(sum(mean_resr_score.values())) / len(mean_resr_score))
				if weight == 0:
					continue
				score_q = Queue.PriorityQueue()
				for job in self.uncompleted_jobs:
					score = mean_align_score[job] + weight * mean_resr_score[job]
					score_q.put((-score, job))
				while not score_q.empty():
					_, job = score_q.get()
					if job.num_workers >= pm.MAX_NUM_WORKERS:
						continue
					else:
						# alloc resr
						if pm.PS_WORKER:
							resr_reqs = job.resr_worker + job.resr_ps
						else:
							resr_reqs = job.resr_worker
						succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
						if succ:
							if pm.PS_WORKER and pm.BUNDLE_ACTION:
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
								job.dom_share = np.max(1.0 * (
									job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)

								if pm.PS_WORKER:
									self._state(job.id, "ps")
									job.num_ps += 1
									job.curr_ps_placement.append(node)
									job.dom_share = np.max(1.0 * (
										job.num_workers * job.resr_worker + job.num_ps * job.resr_ps) / self.cluster.CLUSTER_RESR_CAPS)
							self.running_jobs.add(job)
							node_used_resr_queue.put((node, node_used_resrs))  # this code must be here instead of above
							break
						else: # fail to alloc resources
							# continue
							break

		toc = time.time()
		self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
		for job in self.uncompleted_jobs:
			self.logger.debug(self.name + ":: scheduling results" +" num_worker: " + str(job.num_workers))


def test():
	import log, trace
	logger = log.getLogger(name="test.log", level="DEBUG")
	job_trace = trace.Trace(logger).get_trace()
	env = Tetris_Env("Tetris", job_trace, logger)
	while not env.end:
		env.step()
	print env.get_results()





if __name__ == '__main__':
	test()





