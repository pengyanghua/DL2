import os
import sys
import params_template as pm
import datetime
import copy
import time
import numpy as np
import os.path
import multiprocessing

# default sl hyper-parameters configuration
sl_config_dict = {"TRAINING_MODE":"SL", "VALUE_NET":False, \
				  "POLICY_NN_MODEL":None, "VALUE_NN_MODEL":None, "CHECKPOINT_INTERVAL":50, \
				  "LEARNING_RATE":0.005, "TOT_NUM_STEPS":200, "VAL_INTERVAL":50, \
				  "NUM_TS_PER_UPDATE":5, "JOB_ORDER_SHUFFLE":True}
NUM_TEST = 5
PARALLELISM = 10
TASK_ID = -1

def replace_params(map, dir):
	pm_md = globals().get('pm', None)
	train_config = dict()
	if pm_md:
		train_config = {key: value for key, value in pm_md.__dict__.iteritems() if not (key.startswith('__') or key.startswith('_'))}

	f = open(dir+"parameters.py", 'w')
	for key, _ in train_config.iteritems():
		if key in map.keys():
			train_config[key] = map[key]
		if isinstance(train_config[key], basestring):
			f.write(str(key) + " = " + "'" + str(train_config[key]) + "'" + '\n')
		else:
			f.write(str(key) + " = " + str(train_config[key])+'\n')
	f.close()


def get_config(id, exp_name, test_value):
	config = dict()
	config["EXPERIMENT_NAME"] = exp_name + "_" + str(test_value)
	if id == 1:
		config["SCHED_WINDOW_SIZE"] = test_value
		config["STATE_DIM"] = (sum([enable for (_,enable) in pm.INPUTS_GATE]), test_value)
		config["ACTION_DIM"] = 3 * test_value + pm.SKIP_TS
		config["NUM_NEURONS_PER_FCN"] = sum([enable for (_,enable) in pm.INPUTS_GATE]) * test_value
	elif id == 2:
		config["NUM_FCN_LAYERS"] = 1
		config["NUM_NEURONS_PER_FCN"] = test_value
	elif id == 3 or id == 24:
		config["NUM_FCN_LAYERS"] = test_value
		config["NUM_NEURONS_PER_FCN"] = pm.STATE_DIM[0]*pm.STATE_DIM[1]*2/3
	elif id == 4:
		config["BUNDLE_ACTION"] = test_value
		if test_value == False:
			config["ACTION_DIM"] = 2 * pm.SCHED_WINDOW_SIZE + pm.SKIP_TS
	elif id == 5:
		config["JOB_ARRIVAL_PATTERN"] = test_value
	elif id == 6:
		config["BATCH_NORMALIZATION"] = test_value
	elif id == 7:
		config["SL_LOSS_FUNCTION"] = test_value
	elif id == 8:
		["Norm_Progress", "Job_Progress", "Num_Uncompleted_Jobs"]
		if test_value == "Norm_Progress":
			config["TS_REWARD_PLUS_JOB_REWARD"] = False
			config["NUM_UNCOMPLETED_JOB_REWARD"] = False
		elif test_value == "Job_Progress":
			config["TS_REWARD_PLUS_JOB_REWARD"] = True
			config["NUM_UNCOMPLETED_JOB_REWARD"] = False
		elif test_value == "Num_Uncompleted_Jobs":
			config["TS_REWARD_PLUS_JOB_REWARD"] = False
			config["NUM_UNCOMPLETED_JOB_REWARD"] = True
	elif id == 9:
		if not test_value:
			config["REPLAY_MEMORY_SIZE"] = 256
	elif id == 10:
		config["VALUE_NET"] = test_value
	elif id == 11:
		if test_value:
			config["INJECT_SAMPLES"] = True
			config["EPSILON_GREEDY"] = False
		else:
			config["INJECT_SAMPLES"] = False
			config["EPSILON_GREEDY"] = True
	elif id == 12:
		config["JOB_ARRIVAL_PATTERN"] = test_value
		config["HEURISTIC"] = "DRF"
	elif id == 13:
		config["JOB_ARRIVAL_PATTERN"] = test_value
		config["HEURISTIC"] = "SRTF"
	elif id == 14:
		config["JOB_ARRIVAL_PATTERN"] = test_value
		config["HEURISTIC"] = "Tetris"
	elif id == 15:
		config["JOB_ARRIVAL_PATTERN"] = test_value
		config["HEURISTIC"] = "Optimus"
	elif id == 16:
		config["HEURISTIC"] = test_value
		config["MAX_NUM_WORKERS"] = 8
	elif id == 17:
		config["NUM_AGENTS"] = test_value
		config["MINI_BATCH_SIZE"] = 256/test_value
	elif id == 18:
		config["CHANGING_JOB_TYPES"] = test_value
	elif id == 19:
		config["REAL_SPEED_TRACE"] = test_value
	elif id == 20:
		if test_value == "testbed":
			config["TESTBED"] = True
			config["CLUSTER_NUM_NODES"] = 6
			config["TOT_NUM_JOBS"] = 10
			config["MAX_NUM_EPOCHS"] = 1000
			config["MAX_ARRVS_PER_TS"] = 5
			config["TS_DURATION"] = 300.0
			window_size = 4
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "large-1":
			config["LARGE_SCALE"] = True
			config["CLUSTER_NUM_NODES"] = 100
			config["TOT_NUM_JOBS"] = 120
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 6
			config["TS_DURATION"] = 1200.0
			window_size = 30
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "large-2":
			config["LARGE_SCALE"] = True
			config["CLUSTER_NUM_NODES"] = 100
			config["TOT_NUM_JOBS"] = 180
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 9
			config["TS_DURATION"] = 1200.0
			window_size = 36
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "large-3":
			config["LARGE_SCALE"] = True
			config["CLUSTER_NUM_NODES"] = 120
			config["TOT_NUM_JOBS"] = 180
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 9
			config["TS_DURATION"] = 1200.0
			window_size = 36
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "large-4":
			config["LARGE_SCALE"] = True
			config["CLUSTER_NUM_NODES"] = 500
			config["TOT_NUM_JOBS"] = 600
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 30
			config["TS_DURATION"] = 1200.0
			config["MAX_NUM_WORKERS"] = 50
			window_size = 180
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "large-5":
			config["LARGE_SCALE"] = True
			config["CLUSTER_NUM_NODES"] = 500
			config["TOT_NUM_JOBS"] = 600
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 30
			config["TS_DURATION"] = 1200.0
			config["MAX_NUM_WORKERS"] = 100
			window_size = 180
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "large-6":
			config["LARGE_SCALE"] = True
			config["CLUSTER_NUM_NODES"] = 500
			config["TOT_NUM_JOBS"] = 600
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 30
			config["TS_DURATION"] = 1200.0
			config["MAX_NUM_WORKERS"] = 100
			config["VALUE_NET"] = False
			window_size = 180
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
		elif test_value == "small": # by default
			config["CLUSTER_NUM_NODES"] = 48
			config["TOT_NUM_JOBS"] = 60
			config["MAX_NUM_EPOCHS"] = 80000
			config["MAX_ARRVS_PER_TS"] = 3
			config["TS_DURATION"] = 1200.0
			window_size = 20
			config["SCHED_WINDOW_SIZE"] = window_size
			config["STATE_DIM"] = (sum([enable for (_, enable) in pm.INPUTS_GATE]), window_size)
			config["ACTION_DIM"] = 3 * window_size + pm.SKIP_TS
			config["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pm.INPUTS_GATE]) * window_size
	elif id == 21:
		config["JOB_RESR_BALANCE"] = test_value
	elif id == 22:
		if not test_value:
			config["POLICY_NN_MODEL"] = None
	elif id == 23:
		config["JOB_EPOCH_EST_ERROR"] = test_value
	elif id == 25:
		config["TRAIN_SPEED_ERROR"] = test_value
	return config


def process_results(root_dir, exp_name, test_values):
	results = dict()
	for test_value in test_values:
		jcts = []
		makespans = []
		rewards = []
		for j in range(NUM_TEST):
			dir = root_dir + exp_name + "_" + str(test_value) + "/" + str(j) + '/'
			file = dir+exp_name+"_"+str(test_value)+"/rl_validation.txt"
			assert os.path.exists(file)
			f = open(file, 'r')
			temp_jcts = []
			temp_makespans = []
			temp_rewards = []
			for line in f:
				segs = line.replace("\n",'').split(" ")
				temp_jcts.append(float(segs[2]))
				temp_makespans.append(float(segs[3]))
				temp_rewards.append(float(segs[4]))
			# find the min jct
			min_index = np.argmin(temp_jcts)
			jcts.append(temp_jcts[min_index])
			makespans.append(temp_makespans[min_index])
			rewards.append(temp_rewards[min_index])
		results[test_value] = (str(np.average(jcts))+"+-"+str(np.std(jcts)),\
							   str(np.average(makespans))+"+-"+str(np.std(makespans)),\
							   str(np.average(rewards))+"+-"+str(np.std(rewards)))
	f = open(root_dir+"results.txt", "w")
	for item in results.items():
		f.write(str(item) + "\n")
	f.close()
	print results
	return results


def _sl_rl(dir, config, device):
	# SL
	sl_config = copy.deepcopy(sl_config_dict)
	for key, value in config.items():
		if key not in sl_config:  # sl_config_dict has higher priority
			sl_config[key] = value
	os.system("mkdir -p " + dir)
	os.system("cp *.py *.txt " + dir)
	replace_params(sl_config, dir)
	if TASK_ID != 17:
		os.system("cd " + dir + " && CUDA_VISIBLE_DEVICES=" + str(device) + " python train.py")
	else:
		os.system("cd " + dir + " && python train.py")

	time.sleep(3)
	# RL
	replace_params(config, dir)
	if TASK_ID != 17:
		os.system("cd " + dir + " && CUDA_VISIBLE_DEVICES=" + str(device) + " python train.py")
	else:
		os.system("cd " + dir + " && python train.py")


def _baseline(dir, config):
	os.system("mkdir -p " + dir)
	os.system("cp *.py *.txt " + dir)
	replace_params(config, dir)
	os.system("cd " + dir + " && python comparison.py")


def run(id, exp_name, test_values):
	print "running experiments for", exp_name
	tic = time.time()
	root_dir = exp_name + "-" + datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "/"

	pool = multiprocessing.Pool(processes=PARALLELISM)
	for i in range(len(test_values)):
		test_value = test_values[i]
		print "testing", exp_name, "with value", test_value
		parent_dir = root_dir + exp_name + "_" + str(test_value) + "/"
		for j in range(NUM_TEST):
			print "round", j
			dir = parent_dir + str(j) + '/'
			config = get_config(id, exp_name, test_value)
			device = (i*NUM_TEST+j)%2
			if id in [12, 13, 14, 15]:
				# _baseline(dir, config)
				pool.apply_async(_baseline, args=(dir, config))
			else:
				# _sl_rl(dir, config, device)
				pool.apply_async(_sl_rl, args=(dir, config, device))
			if id in [12, 13, 14, 15]:
				time.sleep(0.3)
			else:
				time.sleep(3)

	pool.close()
	pool.join()

	results = process_results(root_dir, exp_name, test_values)
	print "finish testing all values of", exp_name
	print "the result is:", results
	toc = time.time()
	print "elapsed time: ", toc - tic, "seconds"



def main(id):
	global PARALLELISM, TASK_ID
	TASK_ID = id
	if id == 1:
		exp_name = "sched_window_size"
		test_values = [10, 20, 30, 40, 50, 60]
	elif id == 2:
		exp_name = "number_of_neurons"
		test_values = [16, 32, 64, 96, 128, 160, 192, 256]
	elif id == 3:
		PARALLELISM = 5
		exp_name = "number_of_hidden_layers"
		test_values = [1, 2, 3, 4]
	elif id == 4:
		exp_name = "bundle_action" # bundle false error
		test_values = [False, True]
	elif id == 5:
		exp_name = "job_arrival_distribution"
		test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
	elif id == 6:
		exp_name = "batch_normalization"
		test_values = [False, True]
	elif id == 7:
		exp_name = "sl_loss_function"
		test_values = ["Mean_Square", "Cross_Entropy", "Absolute_Difference"]
	elif id == 8:
		exp_name = "job_reward_function"
		test_values = ["Norm_Progress", "Job_Progress", "Num_Uncompleted_Jobs"]
	elif id == 9:
		exp_name = "experience_replay"
		test_values = [False, True]
	elif id == 10:
		exp_name = "critic_network"
		test_values = [False, True]
	elif id == 11:
		exp_name = "exploration"
		test_values = [False, True]
	elif id == 12:
		exp_name = "DRF_baseline"
		test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
	elif id == 13:
		exp_name = "SRTF_baseline"
		test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
	elif id == 14:
		exp_name = "Tetris_baseline"
		test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
	elif id == 15:
		exp_name = "Optimus_baseline"
		test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
	elif id == 16:
		exp_name = "SL_heuristics"
		test_values = ["Optimus", "FIFO", "SRTF"]
	elif id == 17:
		PARALLELISM = 5
		exp_name = "a3c"
		test_values = [5, 4, 3, 2, 1]
	elif id == 18:
		exp_name = "changing_job_types"
		test_values = [True]
	elif id == 19:
		exp_name = "analytical_model"
		test_values = [False]
	elif id == 20:
		exp_name = "cluster_scale"
		test_values = ["large-4", "large-5", "large-6", "large-1", "large-2", "large-3", "testbed", "small"]
	elif id == 21:
		exp_name = "job_resr_balance"
		test_values = [True, False]
	elif id == 22:
		exp_name = "enable_SL_or_not"
		test_values = [True, False]
	elif id == 23:
		exp_name = "estimation_error_num_epoch" # error
		test_values = [0.05, 0.1, 0.15, 0.2, 0.25]
	elif id == 24:
		PARALLELISM = 3
		exp_name = "number_of_hidden_layers"
		test_values = [5, 6, 7]
	elif id == 25:
		exp_name = "train_speed_error"
		test_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

	run(id, exp_name, test_values)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "a script for running experiment"
		print "Usage: please input one of following experiment IDs"
		print "1: scheduling window size"
		print "2: number of neurons"
		print "3: number of hidden layers"
		print "4: bundle action"
		print "5: job arrival distribution"
		print "6: batch normalization"
		print "7: sl loss function"
		print "8: job reward function"
		print "9: experience replay"
		print "10: critic network"
		print "11: exploration"
		print "12: DRF baseline"
		print "13: SRTF baseline"
		print "14: Tetris baseline"
		print "15: Optimus baseline"
		print "16: SL heuristics"
		print "17: a3c, change train_a3c.py to train.py, change parallelism, make sure a correct total batch size before running"
		print "18: changing job types during training"
		print "19: training on analytical model"
		print "20: cluster scale"
		print "21: job resource balance"
		print "22: enable SL or not"
		print "23: estimation error of epoch number"
		print "25: train speed error"
		exit(1)
	main(int(sys.argv[1]))