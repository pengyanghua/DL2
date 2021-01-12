import Queue
import time
import numpy as np
import parameters as pm
from scheduler_base import Scheduler

class DRF_Env(Scheduler):
	# overwrite the scheduling algorithm in Scheduler
	def _schedule(self):
		# DRF
		tic = time.time()

		drf_queue = Queue.PriorityQueue()
		for job in self.uncompleted_jobs:
			# init num_ps and num_worker
			job.num_workers = 0
			job.num_ps = 0
			drf_queue.put((0, job.arrv_time, job))  # enqueue jobs into drf queue

		self.running_jobs = set()

		# keep track of available resources on each node.
		node_used_cpu_list = [0 for i in range(pm.CLUSTER_NUM_NODES)]
		node_used_mem_list = [0 for i in range(pm.CLUSTER_NUM_NODES)]
		node_used_gpu_list = [0 for i in range(pm.CLUSTER_NUM_NODES)]
		node_used_bw_list = [0 for i in range(pm.CLUSTER_NUM_NODES)]

		# cur_node_index = 0
		node_used_resr_queue = Queue.PriorityQueue()
		for i in range(pm.CLUSTER_NUM_NODES):
			node_used_resr_queue.put((0, i))
		placements = dict()  # job_id:placement_list

		while not drf_queue.empty():
			(dom_share, job_arrival, job) = drf_queue.get()
			# bundle one ps and one worker together by default
			cpu_req = job.resr_worker[0] + job.resr_ps[0]
			mem_req = 0  # job.worker_mem + job.ps_mem
			bw_req = 0  # job.worker_bw + job.ps_bw
			gpu_req = job.resr_worker[1] + job.resr_ps[1]

			# check whether resources are sufficient
			print cpu_req, gpu_req
			# node_index = (cur_node_index + i) % len(params.NODE_LIST)  # check all nodes
			_, node_index = node_used_resr_queue.get()
			suff_resr = True
			if node_used_cpu_list[node_index] + cpu_req > 8 or \
									node_used_mem_list[node_index] + mem_req > 48 or \
									node_used_bw_list[node_index] + bw_req > 10 or \
									node_used_gpu_list[node_index] + gpu_req > 8:
				suff_resr = False
			print suff_resr
			if suff_resr:
				job.num_workers += 1
				job.num_ps += 1
				node_used_cpu_list[node_index] += cpu_req
				node_used_mem_list[node_index] += mem_req
				node_used_bw_list[node_index] += bw_req
				node_used_gpu_list[node_index] += gpu_req
				node_used_resr_queue.put((node_used_cpu_list[node_index] + node_used_gpu_list[node_index], node_index))
				# placement
				if job.id in placements:
					placements[job.id].append(node_index)
				else:
					placements[job.id] = [node_index]
				job.curr_ps_placement.append(node_index)
				job.curr_worker_placement.append(node_index)
				# cur_node_index = (node_index + 1) % len(params.NODE_LIST)  # update index if round-robin, otherwise adopt best fit packing

				# update dominant resource
				cpu_share = 1.0 * (job.num_workers * job.resr_worker[0] + job.num_ps * job.resr_ps[0]) / 48
				#mem_share = 1.0 * (job.num_worker * job.worker_mem + job.num_ps * job.ps_mem) / 288
				#bw_share = 1.0 * (job.num_worker * job.worker_bw + job.num_ps * job.ps_bw) / 60
				gpu_share = 1.0 * (job.num_workers * job.resr_worker[1]) / 48
				dom_share = max(cpu_share, gpu_share)
				if job.num_workers < 16 and job.num_ps < 16:
					drf_queue.put((dom_share, job_arrival, job))

				if job not in self.running_jobs:
					self.running_jobs.add(job)
			else:
				self.cluster_used_cpu = sum(node_used_cpu_list)
				self.cluster_used_mem = sum(node_used_mem_list)
				self.cluster_used_bw = sum(node_used_bw_list)
				self.cluster_used_gpu = sum(node_used_gpu_list)
				break  # no enough resource

		toc = time.time()
		self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")

		toc = time.time()
		self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")
		for job in self.uncompleted_jobs:
			self.logger.debug(self.name + ":: scheduling results" + "job id: " + str(job.id) + " num_worker: " + str(job.num_workers) +" num_ps: " + str(job.num_ps))
		a = raw_input()

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





