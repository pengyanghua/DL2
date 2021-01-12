import numpy as np
import time
import parameters as pm
import drf_env
import fifo_env
import tetris_env
import srtf_env
import optimus_env
import rl_env


def val_loss(net, val_traces, logger, global_step):
	avg_loss = 0
	step = 0
	data = []
	for episode in range(len(val_traces)):
		job_trace = val_traces[episode]
		if pm.HEURISTIC == "DRF":
			env = drf_env.DRF_Env("DRF", job_trace, logger)
		elif pm.HEURISTIC == "FIFO":
			env = fifo_env.FIFO_Env("FIFO", job_trace, logger)
		elif pm.HEURISTIC == "SRTF":
			env = srtf_env.SRTF_Env("SRTF", job_trace, logger)
		elif pm.HEURISTIC == "Tetris":
			env = tetris_env.Tetris_Env("Tetris", job_trace, logger)
		elif pm.HEURISTIC == "Optimus":
			env = optimus_env.Optimus_Env("Optimus", job_trace, logger)

		ts = 0
		while not env.end:
			data += env.step()
			ts += 1
			if len(data) >= pm.MINI_BATCH_SIZE:
				# prepare a validation batch
				indexes = np.random.choice(len(data), size=pm.MINI_BATCH_SIZE, replace=False)
				inputs = []
				labels = []
				for index in indexes:
					input, label = data[index]
					inputs.append(input)
					labels.append(label)
				# superversed learning to calculate gradients
				output, loss = net.get_sl_loss(np.stack(inputs),np.vstack(labels))
				avg_loss += loss
				# if step%50 == 0:
				# 	# # type, # of time slots in the system so far, normalized remaining epoch, dom resource
				# 	tb_logger.add_text(tag="sl:input+label+output:" + str(episode) + "_" + str(ts), value="input:" + \
				# 		" type: "+ str(input[0]) + " stay_ts: " + str(input[1]) + " rt: " + str(input[2]) \
				# 		+ " resr:" + str(input[3]) + "\n" +
				# 		" label: " + str(label) + "\n" + " output: " + str(output[-1]), step=global_step)
				step += 1
				data = []

	return avg_loss/step


def val_jmr(net, val_traces, logger, global_step, tb_logger):
	avg_jct = []
	avg_makespan = []
	avg_reward = []
	step = 0.0
	tic = time.time()
	stats = dict()
	stats["step"] = global_step
	stats["jcts"] = []
	states_dict = dict()
	states_dict["step"] = global_step
	states_dict["states"] = []
	for episode in range(len(val_traces)):
		job_trace = val_traces[episode]
		env = rl_env.RL_Env("RL", job_trace, logger, False)
		ts = 0
		while not env.end:
			input = env.observe()
			output = net.predict(np.reshape(input,(1, pm.STATE_DIM[0], pm.STATE_DIM[1])))
			masked_output, action, reward, move_on, valid_state = env.step(output)
			if episode == 0 and move_on: # record the first trace
				states = env.get_sched_states()
				states_dict["states"].append(states)
				'''
				job id: type: num_workers:
				'''
				string = "ts: " + str(ts) + " "
				for id,type,num_workers,num_ps in states:
					if pm.PS_WORKER:
						string += "(id: "+str(id) + " type: " + str(type) + " num_workers: " + str(num_workers) + " num_ps: " + str(num_ps) + ") \n"
					else:
						string += "(id: " + str(id) + " type: " + str(type) + " num_workers: " + str(num_workers) + ") \n"
				tb_logger.add_text(tag="rl:resr_allocation:" + str(episode)+str(global_step), value=string, step=global_step)
				ts += 1

			if episode == 0:
				if step % 50 == 0:
					i = 0
					value = "input:"
					for (key, enabled) in pm.INPUTS_GATE:
						if enabled:
							# [("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",True), ("WORKERS",False)]
							if key == "TYPE":
								value += " type: " + str(input[i]) + "\n\n"
							elif key == "STAY":
								value += " stay_ts: " + str(input[i]) + "\n\n"
							elif key == "PROGRESS":
								value += " rt: " + str(input[i]) + "\n\n"
							elif key == "DOM_RESR":
								value += " resr: " + str(input[i]) + "\n\n"
							elif key == "WORKERS":
								value += " workers: " + str(input[i]) + "\n\n"
							elif key == "PS":
								value += " ps: " + str(input[i]) + "\n\n"
							i += 1
					value += " output: " + str(output) + "\n\n" + " masked_output: " + str(masked_output) + "\n\n" + " action: " + str(action)

					tb_logger.add_text(tag="rl:input+output+action:" + str(global_step) + "_" + str(episode) + "_" + str(ts) + "_" + str(step),
						value=value, step=global_step)
			step += 1
		num_jobs, jct, makespan, reward = env.get_results()
		stats["jcts"].append(env.get_job_jcts().values())
		avg_jct.append(jct)
		avg_makespan.append(makespan)
		avg_reward.append(reward)
	elapsed_t = time.time() - tic
	logger.info("time for making one decision: " + str(elapsed_t / step) + " seconds")
	with open("DL2_JCTs.txt", 'a') as f:
		f.write(str(stats) + '\n')
	with open("DL2_states.txt", 'a') as f:
		f.write(str(states_dict)+"\n")

	return (1.0*sum(avg_jct)/len(avg_jct), 1.0*sum(avg_makespan)/len(avg_makespan), sum(avg_reward)/len(avg_reward))