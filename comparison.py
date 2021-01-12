import parameters
import multiprocessing
import parameters as pm
import os
import log
import trace
import time
import drf_env
import srtf_env
import fifo_env
import tetris_env
import optimus_env
import copy_reg
import types


# register method instance as pickable objects
def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def drf(job_trace=None):
	if job_trace is None:
		job_trace = trace.Trace(None).get_trace()
	env = drf_env.DRF_Env("DRF", job_trace, None)
	while not env.end:
		env.step()
	return [env.get_results(), env.get_job_jcts().values()]


def srtf(job_trace=None):
	if job_trace is None:
		job_trace = trace.Trace(None).get_trace()
	env = srtf_env.SRTF_Env("SRTF", job_trace, None)
	while not env.end:
		env.step()
	return [env.get_results(), env.get_job_jcts().values()]

def fifo(job_trace=None):
	if job_trace is None:
		job_trace = trace.Trace(None).get_trace()
	env = fifo_env.FIFO_Env("FIFO", job_trace, None)
	while not env.end:
		env.step()
	return [env.get_results(), env.get_job_jcts().values()]

def tetris(job_trace=None):
	if job_trace is None:
		job_trace = trace.Trace(None).get_trace()
	env = tetris_env.Tetris_Env("Tetris", job_trace, None)
	while not env.end:
		env.step()
	return [env.get_results(), env.get_job_jcts().values()]

def optimus(job_trace=None):
	if job_trace is None:
		job_trace = trace.Trace(None).get_trace()
	env = optimus_env.Optimus_Env("Optimus", job_trace, None)
	while not env.end:
		env.step()
	return [env.get_results(), env.get_job_jcts().values()]



def compare(traces, logger, debug="False"):
	if debug:
		drf(traces[0])
		srtf(traces[0])
		fifo(traces[0])
		tetris(traces[0])
		optimus(traces[0])
	f = open("DRF_JCTs.txt", 'w')
	f.close()

	num_schedulers = 5
	thread_list = [[] for i in range(num_schedulers)]  # a two dimension matrix
	tic = time.time()
	pool = multiprocessing.Pool(processes=40)
	for i in range(len(traces)):  # one example takes about 10s
		thread_list[0].append(pool.apply_async(drf, args=(traces[i],)))
		thread_list[1].append(pool.apply_async(srtf, args=(traces[i],)))
		thread_list[2].append(pool.apply_async(fifo, args=(traces[i],)))
		thread_list[3].append(pool.apply_async(tetris, args=(traces[i],)))
		thread_list[4].append(pool.apply_async(optimus, args=(traces[i],)))
	pool.close()
	pool.join()

	jct_list = [[] for i in range(num_schedulers)]  # a two dimension matrix
	makespan_list = [[] for i in range(num_schedulers)]
	reward_list = [[] for i in range(num_schedulers)]
	for i in range(num_schedulers):
		for j in range(len(thread_list[i])):
			result, jcts = thread_list[i][j].get()
			if i == 0: # DRF
				with open("DRF_JCTs.txt", 'a') as f:
					f.write(str(jcts)+'\n')
			num_jobs, jct, makespan, reward = result
			jct_list[i].append(jct)
			makespan_list[i].append(makespan)
			reward_list[i].append(reward)
		toc = time.time()

	logger.info("---------------------------------------------------------------")
	logger.info("progress: finish testing " + str(len(traces)) + " traces within " + str(int(toc - tic)) + " seconds")
	logger.info("Average      JCT: DRF " + '%.3f' % (sum(jct_list[0]) / len(jct_list[0])) + " SRTF " + \
				 '%.3f' % (sum(jct_list[1]) / len(jct_list[1])) + " FIFO " + '%.3f' % (sum(jct_list[2]) / len(jct_list[2])) \
				 + " Tetris " + '%.3f' % (sum(jct_list[3]) / len(jct_list[3]))  + " Optimus " + '%.3f' % (sum(jct_list[4]) / len(jct_list[4])))
	logger.info("Average Makespan: DRF " + '%.3f' % (1.0 * sum(makespan_list[0]) / len(makespan_list[0])) + \
				 " SRTF " + '%.3f' % (1.0 * sum(makespan_list[1]) / len(makespan_list[1])) + \
		" FIFO " + '%.3f' % (1.0 * sum(makespan_list[2]) / len(makespan_list[2])) + " Tetris " + '%.3f' % (
				1.0 * sum(makespan_list[3]) / len(makespan_list[3]))   + " Optimus " + '%.3f' % (sum(makespan_list[4]) / len(makespan_list[4])))
	logger.info("Average   Reward: DRF " + '%.3f' % (1.0 * sum(reward_list[0]) / len(reward_list[0])) + \
				 " SRTF " + '%.3f' % (1.0 * sum(reward_list[1]) / len(reward_list[1])) + \
		" FIFO " + '%.3f' % (1.0 * sum(reward_list[2]) / len(reward_list[2])) + " Tetris " + '%.3f' % (
				1.0 * sum(reward_list[3]) / len(reward_list[3]))   + " Optimus " + '%.3f' % (sum(reward_list[4]) / len(reward_list[4])))
	stats = [() for i in range(num_schedulers)]
	for i in range(num_schedulers):
		jct = 1.0*sum(jct_list[i]) / len(jct_list[i])
		makespan = 1.0*sum(makespan_list[i]) / len(makespan_list[i])
		reward = 1.0*sum(reward_list[i]) / len(reward_list[i])
		stats[i] = (jct, makespan, reward)

	if pm.EXPERIMENT_NAME is not None:
		LOG_DIR = "./" + pm.EXPERIMENT_NAME + "/"
		os.system("rm -rf " + LOG_DIR)
		os.system("mkdir -p " + LOG_DIR + "; cp *.py *.txt " + LOG_DIR)
		f = open(LOG_DIR + "rl_validation.txt", 'a')
		tags_prefix = ["DRF", "SRTF", "FIFO", "Tetris", "Optimus"]
		assert len(tags_prefix) == len(stats)
		for i in range(len(stats)):
			if pm.HEURISTIC == tags_prefix[i]:
				jct, makespan, reward = stats[i]
				f.write(pm.HEURISTIC + " 0: " + str(jct) + " " + str(makespan) + " " + str(reward) + "\n")
		f.close()

	return stats



def main():
	logger = log.getLogger(name="comparison", level="INFO")
	num_traces = 10
	traces = []
	for i in range(num_traces):
		job_trace = trace.Trace(None).get_trace()
		traces.append(job_trace)
	compare(traces, logger, False)

if __name__ == '__main__':
	main()


'''
comparison.py:74 INFO: Average      JCT: DRF 5.900 SRTF 8.132 FIFO 8.203 Tetris 9.606
comparison.py:78 INFO: Average Makespan: DRF 29.207 SRTF 36.991 FIFO 37.221 Tetris 36.204
comparison.py:82 INFO: Average   Reward: DRF 2.063 SRTF 1.633 FIFO 1.623 Tetris 1.668
'''