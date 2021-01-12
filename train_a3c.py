import time
import numpy as np
import multiprocessing
import tensorflow as tf
import os
import parameters as pm
import trace
import network
import drf_env
import fifo_env
import srtf_env
import tetris_env
import rl_env
import log
import validate
import collections
import memory
import prioritized_memory
import tb_log
import copy
import comparison


def collect_stats(stats_qs, tb_logger, step):
	policy_entropys = []
	policy_losses = []
	value_losses = []
	td_losses = []
	step_rewards = []
	jcts = []
	makespans = []
	rewards = []
	val_losses = []
	val_jcts = []
	val_makespans = []
	val_rewards = []
	for id in range(pm.NUM_AGENTS):
		while not stats_qs[id].empty():
			stats = stats_qs[id].get()
			tag_prefix = "SAgent " + str(id) + " "
			if stats[0] == "step:sl":
				_, entropy, loss = stats
				policy_entropys.append(entropy)
				policy_losses.append(loss)
				if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
					tb_logger.add_scalar(tag=tag_prefix+"SL Loss", value=loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "SL Entropy", value=entropy, step=step)
			elif stats[0] == "val":
				_, val_loss, jct, makespan, reward = stats
				val_losses.append(val_loss)
				val_jcts.append(jct)
				val_makespans.append(makespan)
				val_rewards.append(reward)
				if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
					tb_logger.add_scalar(tag=tag_prefix+"Val Loss", value=val_loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix+"Val JCT", value=jct, step=step)
					tb_logger.add_scalar(tag=tag_prefix+"Val Makespan", value=makespan, step=step)
					tb_logger.add_scalar(tag=tag_prefix+"Val Reward", value=reward, step=step)
			elif stats[0] == "step:policy":
				_, entropy, loss, td_loss, step_reward, output = stats
				policy_entropys.append(entropy)
				policy_losses.append(loss)
				td_losses.append(td_loss)
				step_rewards.append(step_reward)
				if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
					tb_logger.add_scalar(tag=tag_prefix + "Policy Entropy", value=entropy, step=step)
					tb_logger.add_scalar(tag=tag_prefix+"Policy Loss", value=loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "TD Loss", value=td_loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix+"Step Reward", value=step_reward, step=step)
					tb_logger.add_histogram(tag=tag_prefix+"Output", value=output, step=step)
			elif stats[0] == "step:policy+value":
				_, entropy, policy_loss, value_loss, td_loss, step_reward, output = stats
				policy_entropys.append(entropy)
				policy_losses.append(policy_loss)
				value_losses.append(value_loss)
				td_losses.append(td_loss)
				step_rewards.append(step_reward)
				if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
					tb_logger.add_scalar(tag=tag_prefix + "Policy Entropy", value=entropy, step=step)
					tb_logger.add_scalar(tag=tag_prefix+"Policy Loss", value=policy_loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "Value Loss", value=value_loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "TD Loss", value=td_loss, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "Step Reward", value=step_reward, step=step)
					tb_logger.add_histogram(tag=tag_prefix + "Output", value=output, step=step)
			elif stats[0] == "trace:sched_result":
				_, jct, makespan, reward = stats
				jcts.append(jct)
				makespans.append(makespan)
				rewards.append(reward)
				if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
					tb_logger.add_scalar(tag=tag_prefix + "Avg JCT", value=jct, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "Makespan", value=makespan, step=step)
					tb_logger.add_scalar(tag=tag_prefix + "Reward", value=reward, step=step)
			elif stats[0] == "trace:job_stats":
				_, episode, jobstats = stats
				if id < pm.NUM_RECORD_AGENTS and pm.EXPERIMENT_NAME is None:
					job_stats_tag_prefix = tag_prefix + "Trace " + str(episode) + " Step " + str(step) + " "
					for i in range(len(jobstats["arrival"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Arrival", value=jobstats["arrival"][i], step=i)
					for i in range(len(jobstats["ts_completed"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Ts_completed", value=jobstats["ts_completed"][i], step=i)
					for i in range(len(jobstats["tot_completed"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Tot_completed", value=jobstats["tot_completed"][i], step=i)
					for i in range(len(jobstats["uncompleted"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Uncompleted", value=jobstats["uncompleted"][i], step=i)
					for i in range(len(jobstats["running"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Running", value=jobstats["running"][i], step=i)
					for i in range(len(jobstats["total"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Total jobs", value=jobstats["total"][i], step=i)
					for i in range(len(jobstats["backlog"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "Backlog", value=jobstats["backlog"][i], step=i)
					for i in range(len(jobstats["cpu_util"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "CPU_Util", value=jobstats["cpu_util"][i], step=i)
					for i in range(len(jobstats["gpu_util"])):
						tb_logger.add_scalar(tag=job_stats_tag_prefix + "GPU_Util", value=jobstats["gpu_util"][i], step=i)
					tb_logger.add_histogram(tag=job_stats_tag_prefix + "JCT", value=jobstats["duration"], step=step)

	tag_prefix = "Central "
	if len(policy_entropys) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Policy Entropy", value=sum(policy_entropys) / len(policy_entropys), step=step)
	if len(policy_losses) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Policy Loss", value=sum(policy_losses) / len(policy_losses), step=step)
	if len(value_losses) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Value Loss", value=sum(value_losses) / len(value_losses), step=step)
	if len(td_losses) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "TD Loss / Advantage", value=sum(td_losses) / len(td_losses), step=step)
	if len(step_rewards) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Batch Reward", value=sum(step_rewards) / len(step_rewards), step=step)
	if len(jcts) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "JCT", value=sum(jcts) / len(jcts), step=step)
	if len(makespans) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Makespan", value=sum(makespans) / len(makespans), step=step)
	if len(rewards) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Reward", value=sum(rewards) / len(rewards), step=step)
	if len(val_losses) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Val Loss", value=sum(val_losses) / len(val_losses), step=step)
	if len(val_jcts) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Val JCT", value=sum(val_jcts) / len(val_jcts), step=step)
	if len(val_makespans) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Val Makespan", value=sum(val_makespans) / len(val_makespans), step=step)
	if len(val_rewards) > 0:
		tb_logger.add_scalar(tag=tag_prefix + "Val Reward", value=sum(val_rewards) / len(val_rewards), step=step)
	tb_logger.flush()


def test(policy_net, validation_traces, logger, step, tb_logger):
	val_tic = time.time()
	tag_prefix = "Central "
	try:
		if pm.TRAINING_MODE == "SL":
			val_loss = validate.val_loss(policy_net, copy.deepcopy(validation_traces), logger, step)
			tb_logger.add_scalar(tag=tag_prefix + "Val Loss", value=val_loss, step=step)
		jct, makespan, reward = validate.val_jmr(policy_net, copy.deepcopy(validation_traces), logger, step, tb_logger)
		tb_logger.add_scalar(tag=tag_prefix + "Val JCT", value=jct, step=step)
		tb_logger.add_scalar(tag=tag_prefix + "Val Makespan", value=makespan, step=step)
		tb_logger.add_scalar(tag=tag_prefix + "Val Reward", value=reward, step=step)
		tb_logger.flush()
		val_toc = time.time()
		logger.info("Central Agent:" + " Validation at step " + str(step) + " Time: " + '%.3f' % (val_toc - val_tic))

		# log results
		f = open(LOG_DIR + "rl_validation.txt", 'a')
		f.write("step " + str(step) + ": " + str(jct) + " " + str(makespan) + " " + str(reward) + "\n")
		f.close()

		return (jct, makespan, reward)
	except Exception as e:
		logger.error("Error when validation! " + str(e))
		tb_logger.add_text(tag="validation error", value=str(e), step=step)


def log_config(tb_logger):
	# log all configurations in parameters and backup py
	global LOG_DIR
	if pm.EXPERIMENT_NAME is None:
		LOG_DIR = "./backup/"
	else:
		LOG_DIR = "./" + pm.EXPERIMENT_NAME + "/"

	os.system("rm -rf " + LOG_DIR)
	os.system("mkdir -p " + LOG_DIR + "; cp *.py *.txt " + LOG_DIR)

	pm_md = globals().get('pm', None)
	train_config = dict()
	if pm_md:
		train_config = {key: value for key, value in pm_md.__dict__.iteritems() if not (key.startswith('__') or key.startswith('_'))}
	train_config_str = ""
	for key, value in train_config.iteritems():
		train_config_str += "{:<30}{:<100}".format(key, value) + "\n\n"

	tb_logger.add_text(tag="Config", value=train_config_str, step=0)
	tb_logger.flush()

	if pm.TRAINING_MODE == "SL":
		f = open(pm.MODEL_DIR + "sl_model.config", "w")
	else:
		f = open(pm.MODEL_DIR + "rl_model.config", "w")
	f.write(train_config_str)
	f.close()

	f = open(LOG_DIR + "config.md", 'w')
	f.write(train_config_str)
	f.close()


def central_agent(net_weights_qs, net_gradients_qs, stats_qs):
	logger = log.getLogger(name="central_agent", level=pm.LOG_MODE)
	logger.info("Start central agent...")

	if not pm.RANDOMNESS:
		np.random.seed(pm.np_seed)
		tf.set_random_seed(pm.tf_seed)

	config = tf.ConfigProto()
	config.allow_soft_placement=False
	config.gpu_options.allow_growth = True
	tb_logger = tb_log.Logger(pm.SUMMARY_DIR)
	log_config(tb_logger)

	with tf.Session(config=config) as sess:
		policy_net = network.PolicyNetwork(sess, "policy_net", pm.TRAINING_MODE, logger)
		if pm.VALUE_NET:
			value_net = network.ValueNetwork(sess, "value_net", pm.TRAINING_MODE, logger)
		logger.info("Create the policy network, with "+str(policy_net.get_num_weights())+" parameters")

		sess.run(tf.global_variables_initializer())
		tb_logger.add_graph(sess.graph)
		tb_logger.flush()
		policy_tf_saver = tf.train.Saver(max_to_keep=pm.MAX_NUM_CHECKPOINTS, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_net'))
		if pm.POLICY_NN_MODEL is not None:
			policy_tf_saver.restore(sess, pm.POLICY_NN_MODEL)
			logger.info("Policy model "+pm.POLICY_NN_MODEL+" is restored.")

		if pm.VALUE_NET:
			value_tf_saver = tf.train.Saver(max_to_keep=pm.MAX_NUM_CHECKPOINTS, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value_net'))
			if pm.VALUE_NN_MODEL is not None:
				value_tf_saver.restore(sess, pm.VALUE_NN_MODEL)
				logger.info("Value model " + pm.VALUE_NN_MODEL + " is restored.")

		step = 1
		start_t = time.time()

		if pm.VAL_ON_MASTER:
			validation_traces = []  # validation traces
			tags_prefix = ["DRF: ", "SRTF: ", "FIFO: ", "Tetris: ", "Optimus: "]
			for i in range(pm.VAL_DATASET):
				validation_traces.append(trace.Trace(None).get_trace())
			stats = comparison.compare(copy.deepcopy(validation_traces), logger) # deep copy to avoid changes to validation_traces
			if not pm.SKIP_FIRST_VAL:
				stats.append(test(policy_net, copy.deepcopy(validation_traces), logger, step=0, tb_logger=tb_logger))
				tags_prefix.append("Init_NN: ")

			f = open(LOG_DIR + "baselines.txt", 'w')
			for i in range(len(stats)):
				jct, makespan, reward = stats[i]
				value = "JCT: " + str(jct) + " Makespan: " + str(makespan) + " Reward: " + str(reward) + "\n"
				f.write(value)
				tb_logger.add_text(tag=tags_prefix[i], value=value, step=step)
			f.close()
			tb_logger.flush()
			logger.info("Finish validation for heuristics and initialized NN.")

		updated_agents = [] # updated agents in async, will change each time after centeral agent get gradients
		for i in range(pm.NUM_AGENTS):
			updated_agents.append(i)

		while step <= pm.TOT_NUM_STEPS:
			# send updated parameters to agents
			policy_weights = policy_net.get_weights()
			if pm.VALUE_NET:
				value_weights = value_net.get_weights()
				for i in updated_agents:
					assert net_weights_qs[i].qsize() == 0
					net_weights_qs[i].put((policy_weights, value_weights))
			else:# only put weights for the updated agents
				for i in updated_agents:
					assert net_weights_qs[i].qsize() == 0
					net_weights_qs[i].put(policy_weights)
			updated_agents[:] = []
			# display speed
			if step % pm.DISP_INTERVAL == 0:
				elaps_t = time.time() - start_t
				speed = step / elaps_t
				logger.info("Central agent: Step " + str(
					step) + " Speed " + '%.3f' % speed + " batches/sec" + " Time " + '%.3f' % elaps_t + " seconds")


			# statistics
			if pm.TRAINING_MODE == "RL":
				policy_net.anneal_entropy_weight(step)
				tb_logger.add_scalar(tag="Entropy Weight", value=policy_net.entropy_weight, step=step)
				if pm.EPSILON_GREEDY:
					eps = 2 / (1 + np.exp(step / pm.ANNEALING_TEMPERATURE)) * 0.6
					tb_logger.add_scalar(tag="Epsilon Greedy", value=eps, step=step)

			collect_stats(stats_qs, tb_logger, step)
			if not pm.FIX_LEARNING_RATE:
				if step in pm.ADJUST_LR_STEPS:
					policy_net.lr /= 2
					if pm.VALUE_NET:
						value_net.lr /= 2
					logger.info("Learning rate is decreased to " + str(policy_net.lr) + " at step " + str(step))
			if step < pm.STEP_TRAIN_CRITIC_NET:  # set policy net lr to 0 to train critic net only
				policy_net.lr = 0.0

			if step % pm.DISP_INTERVAL == 0:
				tb_logger.add_scalar(tag="Learning rate", value=policy_net.lr, step=step)

			# save model
			if step % pm.CHECKPOINT_INTERVAL == 0:
				name_prefix = ""
				if pm.TRAINING_MODE == "SL":
					name_prefix += "sl_"
				else:
					name_prefix += "rl_"
				if pm.PS_WORKER:
					name_prefix += "ps_worker_"
				else:
					name_prefix += "worker_"

				model_name = pm.MODEL_DIR + "policy_" + name_prefix + str(step) + ".ckpt"
				path = policy_tf_saver.save(sess, model_name)
				logger.info("Policy model saved: " + path)
				if pm.VALUE_NET and pm.SAVE_VALUE_MODEL:
					model_name = pm.MODEL_DIR + "value_" + name_prefix + str(step) + ".ckpt"
					path = value_tf_saver.save(sess, model_name)
					logger.info("Value model saved: " + path)

			# validation
			if pm.VAL_ON_MASTER and step % pm.VAL_INTERVAL == 0:
				test(policy_net, copy.deepcopy(validation_traces), logger, step, tb_logger)

			# poll and update parameters
			# only calc gradients once one queue is not empty
			while True:
				for i in range(0, pm.NUM_AGENTS):
					if net_gradients_qs[i].qsize() == 1:
						updated_agents.append(i)
						if pm.VALUE_NET:
							policy_gradients, value_gradients = net_gradients_qs[i].get()
							value_net.apply_gradients(value_gradients)
							assert len(value_weights) == len(value_gradients)
						else:
							policy_gradients = net_gradients_qs[i].get()
						policy_net.apply_gradients(policy_gradients)
						assert len(policy_weights) == len(policy_gradients)
				if len(updated_agents) > 0:
					break
					# break when obtaining at least one agent's push
			# poll_ids = set([i for i in range(pm.NUM_AGENTS)])
			# avg_policy_grads = []
			# avg_value_grads = []
			# while True:
			# 	for i in poll_ids.copy():
			# 		try:
			# 			if pm.VALUE_NET:
			# 				policy_gradients, value_gradients = net_gradients_qs[i].get(False)
			# 			else:
			# 				policy_gradients = net_gradients_qs[i].get(False)
			# 			poll_ids.remove(i)
			# 			if len(avg_policy_grads) == 0:
			# 				avg_policy_grads = policy_gradients
			# 			else:
			# 				for j in range(len(avg_policy_grads)):
			# 					avg_policy_grads[j] += policy_gradients[j]
			# 			if pm.VALUE_NET:
			# 				if len(avg_value_grads) == 0:
			# 					avg_value_grads = value_gradients
			# 				else:
			# 					for j in range(len(avg_value_grads)):
			# 						avg_value_grads[j] += value_gradients[j]
			# 		except:
			# 			continue
			# 	if len(poll_ids) == 0:
			# 		break
			# for i in range(0, len(avg_policy_grads)):
			# 	avg_policy_grads[i] = avg_policy_grads[i] / pm.NUM_AGENTS
			# policy_net.apply_gradients(avg_policy_grads)
			#
			# if pm.VALUE_NET:
			# 	for i in range(0, len(avg_value_grads)):
			# 		avg_value_grads[i] = avg_value_grads[i] / pm.NUM_AGENTS
			# 	value_net.apply_gradients(avg_value_grads)

			# visualize gradients and weights
			if step % pm.VISUAL_GW_INTERVAL == 0 and pm.EXPERIMENT_NAME is None:
				assert len(policy_weights) == len(policy_gradients)
				for i in range(0,len(policy_weights),10):
					tb_logger.add_histogram(tag="Policy weights " + str(i), value=policy_weights[i], step=step)
					tb_logger.add_histogram(tag="Policy gradients " + str(i), value=policy_gradients[i], step=step)
				if pm.VALUE_NET:
					assert len(value_weights) == len(value_gradients)
					for i in range(0,len(value_weights),10):
						tb_logger.add_histogram(tag="Value weights " + str(i), value=value_weights[i], step=step)
						tb_logger.add_histogram(tag="Value gradients " + str(i), value=value_gradients[i], step=step)

			step += 1

		logger.info("Training ends...")
		if pm.VALUE_NET:
			for i in range(pm.NUM_AGENTS):
				net_weights_qs[i].put(("exit", "exit"))
		else:
			for i in range(pm.NUM_AGENTS):
				net_weights_qs[i].put("exit")
		# os.system("sudo pkill -9 python")
		exit(0)


def sl_agent(net_weights_q, net_gradients_q, stats_q, id):
	logger = log.getLogger(name="agent_"+str(id), level=pm.LOG_MODE)
	logger.info("Start supervised learning, agent " + str(id) + " ...")

	if not pm.RANDOMNESS:
		np.random.seed(pm.np_seed+id+1)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess, tf.device("/gpu:"+str(id%2)):
		policy_net = network.PolicyNetwork(sess, "policy_net", pm.TRAINING_MODE, logger)
		sess.run(tf.global_variables_initializer())  # to avoid batch normalization error

		global_step = 1
		avg_jct = []
		avg_makespan = []
		avg_reward = []
		if not pm.VAL_ON_MASTER:
			validation_traces = []  # validation traces
			for i in range(pm.VAL_DATASET):
				validation_traces.append(trace.Trace(None).get_trace())
		# generate training traces
		traces = []
		for episode in range(pm.TRAIN_EPOCH_SIZE):
			job_trace = trace.Trace(None).get_trace()
			traces.append(job_trace)
		mem_store = memory.Memory(maxlen=pm.REPLAY_MEMORY_SIZE)
		logger.info("Filling experience buffer...")
		for epoch in range(pm.TOT_TRAIN_EPOCHS):
			for episode in range(pm.TRAIN_EPOCH_SIZE):
				tic = time.time()
				job_trace = copy.deepcopy(traces[episode])
				if pm.HEURISTIC == "DRF":
					env = drf_env.DRF_Env("DRF", job_trace, logger)
				elif pm.HEURISTIC == "FIFO":
					env = fifo_env.FIFO_Env("FIFO", job_trace, logger)
				elif pm.HEURISTIC == "SRTF":
					env = srtf_env.SRTF_Env("SRTF", job_trace, logger)
				elif pm.HEURISTIC == "Tetris":
					env = tetris_env.Tetris_Env("Tetris", job_trace, logger)

				while not env.end:
					if pm.LOG_MODE == "DEBUG":
						time.sleep(0.01)
					data = env.step()
					logger.debug("ts length:" + str(len(data)))

					for (input, label) in data:
						mem_store.store(input, 0, label, 0)

					if mem_store.full():
						# prepare a training batch
						_, trajectories, _ = mem_store.sample(pm.MINI_BATCH_SIZE)
						input_batch = [traj.state for traj in trajectories]
						label_batch = [traj.action for traj in trajectories]

						# if global_step % 10 == 0:
						# 	print "input", input_batch[0]
						# 	print "label", label_batch[0]

						# pull latest weights before training
						weights = net_weights_q.get()
						if isinstance(weights, basestring) and weights == "exit":
							logger.info("Agent " + str(id) + " exits.")
							exit(0)
						policy_net.set_weights(weights)

						# superversed learning to calculate gradients
						entropy, loss, policy_grads = policy_net.get_sl_gradients(np.stack(input_batch),np.vstack(label_batch))
						for i in range(len(policy_grads)):
							assert np.any(np.isnan(policy_grads[i])) == False

						# send gradients to the central agent
						net_gradients_q.put(policy_grads)

						# validation
						if not pm.VAL_ON_MASTER and global_step % pm.VAL_INTERVAL == 0:
							val_tic = time.time()
							val_loss = validate.val_loss(policy_net, validation_traces, logger, global_step)
							jct, makespan, reward = validate.val_jmr(policy_net, validation_traces, logger, global_step)
							stats_q.put(("val", val_loss, jct, makespan, reward))
							val_toc = time.time()
							logger.info("Agent " + str(id) + " Validation at step " + str(global_step) + " Time: " + '%.3f'%(val_toc-val_tic))
						stats_q.put(("step:sl", entropy, loss))

						global_step += 1

				num_jobs, jct, makespan, reward = env.get_results()
				avg_jct.append(jct)
				avg_makespan.append(makespan)
				avg_reward.append(reward)
				if global_step%pm.DISP_INTERVAL == 0:
					logger.info("Agent\t AVG JCT\t Makespan\t Reward")
					logger.info(str(id) + " \t \t " + '%.3f' %(sum(avg_jct)/len(avg_jct)) + " \t\t" + " " + '%.3f' %(1.0*sum(avg_makespan)/len(avg_makespan)) \
								+ " \t" + " " + '%.3f' %(sum(avg_reward)/len(avg_reward)))



def rl_agent(net_weights_q, net_gradients_q, stats_q, id):
	logger = log.getLogger(name="agent_"+str(id), level=pm.LOG_MODE,mode="w",fh=True,ch=True,prefix="Agent " +str(id))
	logger.info("Start reinforcement learning, agent " + str(id) + " ...")

	if not pm.RANDOMNESS:
		np.random.seed(pm.np_seed+id+1)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess, tf.device("/gpu:"+str(id%2)):
		policy_net = network.PolicyNetwork(sess, "policy_net", pm.TRAINING_MODE, logger)
		if pm.VALUE_NET:
			value_net = network.ValueNetwork(sess, "value_net", pm.TRAINING_MODE, logger)
		sess.run(tf.global_variables_initializer())  # to avoid batch normalization error
		if pm.VALUE_NET:
			policy_weights, value_weights = net_weights_q.get()
			value_net.set_weights(value_weights)
		else:
			policy_weights = net_weights_q.get()
		policy_net.set_weights(policy_weights) # initialization from master
		first_time = True

		global_step = 1
		if not pm.VAL_ON_MASTER:
			validation_traces = []
			for i in range(pm.VAL_DATASET):
				validation_traces.append(trace.Trace(None).get_trace())
		if pm.PRIORITY_REPLAY:
			mem_store = prioritized_memory.Memory(maxlen=pm.REPLAY_MEMORY_SIZE)
		else:
			mem_store = memory.Memory(maxlen=pm.REPLAY_MEMORY_SIZE)
		logger.info("Filling experience buffer...")

		# generate training data
		traces = []
		for episode in range(pm.TRAIN_EPOCH_SIZE):
			job_trace = trace.Trace(None).get_trace()
			traces.append(job_trace)

		if pm.EPSILON_GREEDY:
			if pm.VARYING_EPSILON:
				temperature = pm.ANNEALING_TEMPERATURE * (1 + float(id)/pm.NUM_AGENTS)
			else:
				temperature = pm.ANNEALING_TEMPERATURE

		for epoch in range(pm.TOT_TRAIN_EPOCHS):
			for episode in range(pm.TRAIN_EPOCH_SIZE):
				tic = time.time()
				env = rl_env.RL_Env("RL", copy.deepcopy(traces[episode]), logger)

				states = []
				masked_outputs = []
				actions = []
				rewards = []
				ts = 0
				while not env.end:
					if pm.LOG_MODE == "DEBUG":
						time.sleep(0.01)
					state = env.observe()
					output = policy_net.predict(np.reshape(state, (1, pm.STATE_DIM[0], pm.STATE_DIM[1])))
					if pm.EPSILON_GREEDY: # greedy epsilon
						env.epsilon = 2 / (1 + np.exp(global_step / temperature))
					masked_output, action, reward, move_on, valid_state = env.step(output)

					if valid_state: # do not save state when move on except skip_ts, but need to save reward!!!
						states.append(state)
						masked_outputs.append(masked_output)
						actions.append(action)
						rewards.append(reward)
					if move_on:
						ts += 1
						# ts_reward = reward
						if ts%pm.LT_REWARD_NUM_TS == 0 and len(states) > 0: # states can be [] due to no jobs in the ts
							# lt_reward = sum(rewards)
							# ts_rewards = [0 for _ in range(pm.LT_REWARD_NUM_TS)]
							# ts_rewards[-1] = lt_reward
							# for i in reversed(range(0, len(ts_rewards) - 1)):
							# 	ts_rewards[i] += ts_rewards[i + 1] * pm.DISCOUNT_FACTOR

							if pm.LT_REWARD_IN_TS:
								for i in reversed(range(0,len(rewards)-1)):
									rewards[i] += rewards[i+1]*pm.DISCOUNT_FACTOR
							elif pm.TS_REWARD_PLUS_JOB_REWARD:
								rewards = env.get_job_reward()
								assert len(rewards) == len(states)
							else:
								rewards = [reward for _ in range(len(states))]

							# randomly fill samples to memory
							if pm.RANDOM_FILL_MEMORY:
								indexes = np.random.choice(len(states), size=pm.MINI_BATCH_SIZE, replace=False)
								for i in indexes:
									mem_store.store(states[i], masked_outputs[i], actions[i], rewards[i])
							else:
								for i in range(len(states)):
									mem_store.store(states[i], masked_outputs[i], actions[i], rewards[i])

							if mem_store.full() and ts%pm.NUM_TS_PER_UPDATE == 0:
								# print "start training RL"
								# prepare a training batch
								mem_indexes, trajectories, IS_weights = mem_store.sample(pm.MINI_BATCH_SIZE)
								states_batch = [traj.state for traj in trajectories]
								outputs_batch = [traj.output for traj in trajectories]
								actions_batch = [traj.action for traj in trajectories]
								rewards_batch = [traj.reward for traj in trajectories]

								# pull latest weights before training
								if not first_time: # avoid pulling twice at the first update
									if pm.VALUE_NET:
										policy_weights, value_weights = net_weights_q.get()
										if isinstance(policy_weights, basestring) and policy_weights == "exit":
											logger.info("Agent " + str(id) + " exits.")
											exit(0)
										policy_net.set_weights(policy_weights)
										value_net.set_weights(value_weights)
									else:
										policy_weights = net_weights_q.get()
										if isinstance(policy_weights, basestring) and policy_weights == "exit":
											logger.info("Agent " + str(id) + " exits.")
											exit(0)
										policy_net.set_weights(policy_weights)
								else:
									first_time = False

								# set entropy weight, both agent and central agent need to be set
								policy_net.anneal_entropy_weight(global_step)

								# reinforcement learning to calculate gradients
								if pm.VALUE_NET:
									value_output = value_net.predict(np.stack(states_batch))
									td_loss = np.vstack(rewards_batch) - value_output
									adjusted_td_loss = td_loss * np.vstack(IS_weights)
									policy_entropy, policy_loss, policy_grads = policy_net.get_rl_gradients(np.stack(states_batch), \
													np.vstack(outputs_batch), np.vstack(actions_batch), adjusted_td_loss)
									value_loss, value_grads = value_net.get_rl_gradients(np.stack(states_batch), value_output, np.vstack(rewards_batch))
								else:
									if pm.PRIORITY_MEMORY_SORT_REWARD and pm.MEAN_REWARD_BASELINE:
										td_loss = np.vstack(rewards_batch) - mem_store.avg_reward()
									else:
										td_loss = np.vstack(rewards_batch) - 0
									adjusted_td_loss = td_loss * np.vstack(IS_weights)
									policy_entropy, policy_loss, policy_grads = policy_net.get_rl_gradients(np.stack(states_batch), np.vstack(outputs_batch), np.vstack(actions_batch), adjusted_td_loss)

								for aa in range(len(actions_batch)):
									if actions_batch[aa][-1] == 1:
										# print "rewards:", rewards_batch[aa], "td_loss:", td_loss[aa]
										logger.debug("rewards:" + str(rewards_batch[aa]) + "td_loss:" + str(td_loss[aa]))

								for i in range(len(policy_grads)):
									try:
										assert np.any(np.isnan(policy_grads[i])) == False
										# print np.mean(np.abs(policy_grads[i])) # 10^-5 to 10^-2
									except Exception as e:
										logger.error("Error: " + str(e))
										logger.error("Gradients: " + str(policy_grads[i]))
										logger.error("Input type: " + str(states_batch[:,0]))
										logger.error("Masked Output: " + str(outputs_batch))
										logger.error("Action: " + str(actions_batch))
										logger.error("TD Loss: " + str(td_loss))
										logger.error("Policy Loss: " + str(policy_loss))
										logger.error("Policy Entropy: " + str(policy_entropy))
										exit(1) # another option is to continue
								if pm.VALUE_NET:
									for i in range(len(value_grads)):
										try:
											assert np.any(np.isnan(value_grads[i])) == False
										except Exception as e:
											logger.error("Error: " + str(e) + " " + str(policy_grads[i]))
											exit(1)

								# send gradients to the central agent
								if pm.VALUE_NET:
									net_gradients_q.put((policy_grads, value_grads))
								else:
									net_gradients_q.put(policy_grads)
								if pm.PRIORITY_REPLAY:
									mem_store.update(mem_indexes, abs(td_loss))
								# validation
								if not pm.VAL_ON_MASTER and global_step % pm.VAL_INTERVAL == 0:
									val_loss = validate.val_loss(policy_net, validation_traces, logger, global_step)
									jct, makespan, reward = validate.val_jmr(policy_net, validation_traces, logger,
																			 global_step)
									stats_q.put(("val", val_loss, jct, makespan, reward))

								# statistics
								if pm.VALUE_NET:
									stats_q.put(("step:policy+value", policy_entropy, policy_loss, value_loss, sum(td_loss)/len(td_loss), sum(rewards_batch)/len(rewards_batch), output))
								else:
									stats_q.put(("step:policy", policy_entropy, policy_loss, sum(td_loss)/len(td_loss), sum(rewards_batch)/len(rewards_batch), output))
								global_step += 1

							# clear
							states = []
							masked_outputs = []
							actions = []
							rewards = []

				# collect statistics after training one trace
				num_jobs, jct, makespan, reward = env.get_results()
				stats_q.put(("trace:sched_result", jct, makespan, reward))
				if (epoch*pm.TRAIN_EPOCH_SIZE+episode)%pm.DISP_INTERVAL == 0:
					if (epoch*pm.TRAIN_EPOCH_SIZE+episode)%50 == 0:
						stats_q.put(("trace:job_stats", episode, env.get_jobstats()))
					toc = time.time()
					logger.info("--------------------------------------------------------------")
					logger.info("Agent " + str(id) + " Epoch " + str(epoch) + " Trace " + str(episode) + " Step " + str(global_step))
					logger.info("# of Jobs\t AVG JCT\t Makespan\t Reward\t Time")
					logger.info(str(num_jobs) + " \t" + " \t" + " " + '%.3f' %jct + " \t\t" + " " + '%.3f' %makespan \
								+ "\t\t" + " " + '%.3f' %reward + "\t" + " " + '%.3f' % (toc - tic))


def main():
	os.system("rm -f *.log")
	os.system("sudo pkill -9 tensorboard; sleep 3")

	net_weights_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]
	net_gradients_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]
	stats_qs = [multiprocessing.Queue() for i in range(pm.NUM_AGENTS)]

	os.system("mkdir -p " + pm.MODEL_DIR + "; mkdir -p " + pm.SUMMARY_DIR)
	if pm.EXPERIMENT_NAME is None:
		cmd = "cd " + pm.SUMMARY_DIR + " && rm -rf *; tensorboard --logdir=./"
		board = multiprocessing.Process(target=lambda: os.system(cmd), args=())
		board.start()
		time.sleep(3) # let tensorboard start first since it will clear the dir

	# central_agent(net_weights_qs, net_gradients_qs, stats_qs)
	master = multiprocessing.Process(target=central_agent, args=(net_weights_qs, net_gradients_qs, stats_qs,))
	master.start()
	#agent(net_weights_qs[0], net_gradients_qs[0], stats_qs[0], 0)
	#exit()

	if pm.TRAINING_MODE == "SL":
		agents = [multiprocessing.Process(target=sl_agent, args=(net_weights_qs[i], net_gradients_qs[i], stats_qs[i],i,)) for i in range(pm.NUM_AGENTS)]
	elif pm.TRAINING_MODE == "RL":
		agents = [multiprocessing.Process(target=rl_agent, args=(net_weights_qs[i], net_gradients_qs[i], stats_qs[i], i,)) for i in range(pm.NUM_AGENTS)]
	for i in range(pm.NUM_AGENTS):
		agents[i].start()

	master.join()


if __name__ == "__main__":
	main()