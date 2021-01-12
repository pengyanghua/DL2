import Queue
import time
import numpy as np
import parameters as pm
from cluster import Cluster
import log
from scheduler_base import Scheduler


class RL_Env(Scheduler):
    def __init__(self, name, trace, logger, training_mode=True):
        Scheduler.__init__(self, name, trace, logger)

        self.epsilon = 0.0
        self.training_mode = training_mode
        self.sched_seq = []
        self.job_prog_in_ts = dict()
        self.window_jobs = None
        self.jobstats = dict()
        for stats_name in [
                "arrival", "ts_completed", "tot_completed", "duration",
                "uncompleted", "running", "total", "backlog", "cpu_util",
                "gpu_util"
        ]:
            self.jobstats[stats_name] = []
        if pm.PS_WORKER and pm.BUNDLE_ACTION:
            self.action_freq = [0 for _ in range(3)]
        # prepare for the first timeslot
        self._prepare()

    def _prepare(self):
        # admit new jobs
        num_arrv_jobs = 0
        if self.curr_ts in self.trace:
            for job in self.trace[self.curr_ts]:
                job.reset()
                self.uncompleted_jobs.add(job)
                if not self.training_mode:
                    job.training = False
                num_arrv_jobs += 1
                self.logger.debug(job.info())
        self.jobstats["arrival"].append(num_arrv_jobs)
        self.jobstats["total"].append(
            len(self.completed_jobs) + len(self.uncompleted_jobs))
        self.jobstats["backlog"].append(
            max(len(self.uncompleted_jobs) - pm.SCHED_WINDOW_SIZE, 0))

        # reset
        self._sched_states()  # get scheduling states in this ts
        self.running_jobs.clear()
        self.node_used_resr_queue = Queue.PriorityQueue()
        for i in range(pm.CLUSTER_NUM_NODES):
            self.node_used_resr_queue.put((0, i))
        self.cluster.clear()

        for job in self.uncompleted_jobs:
            if pm.ASSIGN_BUNDLE and pm.PS_WORKER:  # assign each job a bundle of ps and worker first to avoid job starvation
                _, node = self.node_used_resr_queue.get()
                resr_reqs = job.resr_worker + job.resr_ps
                succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
                if succ:
                    job.num_ps = 1
                    job.curr_ps_placement = [node]
                    job.num_workers = 1
                    job.curr_worker_placement = [node]
                    job.dom_share = np.max(1.0 *
                                           (job.num_workers * job.resr_worker +
                                            job.num_ps * job.resr_ps) /
                                           self.cluster.CLUSTER_RESR_CAPS)
                    self.running_jobs.add(job)
                else:
                    job.num_workers = 0
                    job.curr_worker_placement = []
                    job.num_ps = 0
                    job.curr_ps_placement = []
                    job.dom_share = 0
                self.node_used_resr_queue.put(
                    (np.sum(node_used_resrs),
                     node))  # always put back to avoid blocking in step()
            else:
                job.num_workers = 0
                job.curr_worker_placement = []
                if pm.PS_WORKER:
                    job.num_ps = 0
                    job.curr_ps_placement = []
                job.dom_share = 0

        if pm.VARYING_SKIP_NUM_WORKERS:
            self.skip_num_workers = np.random.randint(1, pm.MAX_NUM_WORKERS)
        else:
            self.skip_num_workers = 8  #np.random.randint(0,pm.MAX_NUM_WORKERS)
        if pm.VARYING_PS_WORKER_RATIO:
            self.ps_worker_ratio = np.random.randint(3, 8)
        else:
            self.ps_worker_ratio = 5

    def _move(self):
        self._progress()
        if len(self.completed_jobs) == pm.TOT_NUM_JOBS:
            self.end = True
        else:
            # next timeslot
            self.curr_ts += 1
            if self.curr_ts > pm.MAX_TS_LEN:
                self.logger.error(
                    "Exceed the maximal number of timeslot for one trace!")
                self.logger.error("Results: " + str(self.get_results()))
                self.logger.error("Stats: " + str(self.get_jobstats()))
                for job in self.uncompleted_jobs:
                    self.logger.error("Uncompleted job " + str(job.id) +
                                      " tot_epoch: " + str(job.num_epochs) +
                                      " prog: " + str(job.progress) +
                                      " workers: " + str(job.num_workers))
                raise RuntimeError
            self._prepare()

    # step forward by one action
    def step(self, output):
        # mask and adjust probability
        mask = np.ones(pm.ACTION_DIM)
        for i in range(len(self.window_jobs)):
            if self.window_jobs[
                    i] is None:  # what if job workers are already maximum
                if pm.PS_WORKER:
                    if pm.BUNDLE_ACTION:  # worker, ps, bundle
                        mask[3 * i] = 0.0
                        mask[3 * i + 1] = 0.0
                        mask[3 * i + 2] = 0.0
                    else:
                        mask[2 * i] = 0.0
                        mask[2 * i + 1] = 0.0
                else:
                    mask[i] = 0.0
            else:
                if pm.PS_WORKER:
                    worker_full = False
                    ps_full = False
                    if self.window_jobs[i].num_workers >= pm.MAX_NUM_WORKERS:
                        worker_full = True
                    if self.window_jobs[i].num_ps >= pm.MAX_NUM_WORKERS:
                        ps_full = True
                    if worker_full:
                        if pm.BUNDLE_ACTION:
                            mask[3 * i] = 0.0
                        else:
                            mask[2 * i] = 0.0
                    if ps_full:
                        if pm.BUNDLE_ACTION:
                            mask[3 * i + 1] = 0.0
                        else:
                            mask[2 * i + 1] = 0.0
                    if (worker_full or ps_full) and pm.BUNDLE_ACTION:
                        mask[3 * i + 2] = 0.0

        masked_output = np.reshape(output[0] * mask, (1, len(mask)))
        sum_prob = np.sum(masked_output)
        action_vec = np.zeros(len(mask))
        move_on = True
        valid_state = False
        if ((not pm.PS_WORKER) and sum(mask[:len(self.window_jobs)]) == 0) \
          or (pm.PS_WORKER and (not pm.BUNDLE_ACTION) and sum(mask[:2*len(self.window_jobs)]) == 0) \
          or (pm.PS_WORKER and pm.BUNDLE_ACTION and sum(mask[:3*len(self.window_jobs)]) == 0):
            self.logger.debug(
                "All jobs are None, move on and do not save it as a sample")
            self._move()
        elif sum_prob <= 0:
            self.logger.info(
                "All actions are masked or some action with probability 1 is masked!!!"
            )
            if pm.EXPERIMENT_NAME is None:
                self.logger.info(
                    "Output: " + str(output)
                )  # Output: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.  1.  0.]], WHY?
                self.logger.info("Mask: " + str(mask))
                self.logger.info("Window_jobs: " + str(self.window_jobs))
                num_worker_ps_str = ""
                for job in self.window_jobs:
                    if job:
                        num_worker_ps_str += str(job.id) + ": " + str(
                            job.num_ps) + " " + str(job.num_workers) + ","
                self.logger.info("Job: " + num_worker_ps_str)
            self._move()
        else:
            masked_output = masked_output / sum_prob
            if self.training_mode:
                # select action
                if np.random.rand(
                ) > pm.MASK_PROB:  # only valid for training mode
                    masked_output = np.reshape(output[0], (1, len(mask)))
                action_cumsum = np.cumsum(masked_output)
                action = (action_cumsum > np.random.randint(1, pm.RAND_RANGE) /
                          float(pm.RAND_RANGE)).argmax()

                if pm.EPSILON_GREEDY:
                    if np.random.rand() < self.epsilon:
                        val_actions = []
                        for i in range(len(masked_output[0])):
                            if masked_output[0][
                                    i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                val_actions.append(i)
                        action = val_actions[np.random.randint(
                            0, len(val_actions))]

                if pm.INJECT_SAMPLES:
                    if (not pm.REAL_SPEED_TRACE) and (not pm.PS_WORKER):
                        allMaxResr = True
                        for job in self.window_jobs:
                            if job:
                                if job.num_workers > self.skip_num_workers:
                                    continue
                                else:
                                    allMaxResr = False
                                    break
                        if allMaxResr and masked_output[0][len(
                                action_vec
                        ) - 1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                        ) <= pm.SAMPLE_INJECTION_PROB:  # choose to skip if prob larger than a small num, else NaN
                            action = len(action_vec) - 1
                            self.logger.debug("Got 1.")
                    elif pm.REAL_SPEED_TRACE and pm.PS_WORKER:
                        # shuffle = np.random.choice(len(self.window_jobs), len(self.window_jobs), replace=False)  # shuffle is a must, otherwise NN selects only the first several actions!!!
                        if pm.JOB_RESR_BALANCE and pm.BUNDLE_ACTION:
                            max_num_ps_worker = 0
                            min_num_ps_worker = 10**10
                            index_min_job = -1
                            for i in range(len(self.window_jobs)):
                                job = self.window_jobs[i]
                                if job:
                                    num_ps_worker = job.num_ps + job.num_workers
                                    if num_ps_worker > max_num_ps_worker:
                                        max_num_ps_worker = num_ps_worker
                                    if num_ps_worker < min_num_ps_worker:
                                        min_num_ps_worker = num_ps_worker
                                        index_min_job = i
                            if min_num_ps_worker and index_min_job != -1 and max_num_ps_worker / min_num_ps_worker > np.random.randint(
                                    3, 6):
                                if masked_output[0][
                                        3 * index_min_job +
                                        2] > pm.MIN_ACTION_PROB_FOR_SKIP and masked_output[
                                            0][3 *
                                               index_min_job] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                    if np.random.rand() < 0.5:
                                        action = 3 * index_min_job + 2
                                    else:
                                        action = 3 * index_min_job

                        shuffle = [_ for _ in range(len(self.window_jobs))]
                        for i in shuffle:
                            job = self.window_jobs[i]
                            if job:
                                if pm.BUNDLE_ACTION:
                                    # if one of three actions: ps/worker/bundle has low probability, enforce to select it
                                    if min(self.action_freq) > 0 and min(
                                            self.action_freq) * 1.0 / sum(
                                                self.action_freq) < 0.001:
                                        index = np.argmin(self.action_freq)
                                        if mask[3 * i +
                                                index] > 0 and masked_output[0][
                                                    3 * i +
                                                    index] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            action = 3 * i + index
                                            self.logger.debug("Got 0: " +
                                                              str(index))
                                            break
                                    if (job.num_workers == 0
                                            or job.num_ps == 0):
                                        if job.num_ps == 0 and job.num_workers == 0 and mask[
                                                3 * i +
                                                2] > 0 and masked_output[0][
                                                    3 * i +
                                                    2] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                    ) < 0.5:
                                            action = 3 * i + 2
                                            self.logger.debug("Got 1")
                                        if job.num_workers == 0 and mask[
                                                3 *
                                                i] > 0 and masked_output[0][
                                                    3 *
                                                    i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            action = 3 * i
                                        if job.num_ps == 0 and mask[
                                                3 * i +
                                                1] > 0 and masked_output[0][
                                                    3 *
                                                    i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            action = 3 * i + 1
                                        break
                                    elif job.num_ps > job.num_workers * self.ps_worker_ratio and np.random.rand(
                                    ) < 0.5:
                                        if mask[3 * i + 2] > 0 and masked_output[0][
                                                3 * i +
                                                2] > pm.MIN_ACTION_PROB_FOR_SKIP and mask[
                                                    3 *
                                                    i] > 0 and masked_output[0][
                                                        3 *
                                                        i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            if np.random.rand() < 0.5:
                                                # increase this job's bundle
                                                action = 3 * i + 2
                                                self.logger.debug("Got 2.")
                                            else:
                                                action = 3 * i
                                                self.logger.debug("Got 2.")
                                            break
                                    elif job.num_workers >= job.num_ps * 0.5 and np.random.rand(
                                    ) < 0.5:
                                        if mask[3 * i + 2] > 0 and masked_output[0][
                                                3 * i +
                                                2] > pm.MIN_ACTION_PROB_FOR_SKIP and mask[
                                                    3 * i +
                                                    1] > 0 and masked_output[0][
                                                        3 * i +
                                                        1] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            if np.random.rand() < 0.01:
                                                # increase this job's bundle
                                                action = 3 * i + 2
                                                self.logger.debug("Got 3.")
                                            else:
                                                # incrase ps
                                                action = 3 * i + 1
                                                self.logger.debug("Got 4.")
                                            break
                                else:
                                    if job.num_workers == 0 and mask[
                                            2 * i] > 0 and masked_output[0][
                                                2 *
                                                i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.01:
                                        action = 2 * i
                                        self.logger.debug("Got 1.")
                                        break
                                    elif job.num_ps == 0 and mask[
                                            2 * i +
                                            1] > 0 and masked_output[0][
                                                2 * i +
                                                1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.01:
                                        action = 2 * i + 1
                                        self.logger.debug("Got 2.")
                                        break
                                    elif job.num_ps >= job.num_workers * self.ps_worker_ratio and mask[
                                            2 * i] > 0 and masked_output[0][
                                                2 *
                                                i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.5:
                                        # increase this job's worker
                                        action = 2 * i
                                        self.logger.debug("Got 3.")
                                        break
                                    elif job.num_workers >= job.num_ps * self.ps_worker_ratio and mask[
                                            2 * i +
                                            1] > 0 and masked_output[0][
                                                2 * i +
                                                1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.5:
                                        # increase this job's ps
                                        action = 2 * i + 1
                                        self.logger.debug("Got 4.")
                                        break
            else:
                if pm.SELECT_ACTION_MAX_PROB:  # only available for validation
                    action = np.argmax(
                        masked_output
                    )  # output is [[...]] # always select the action with max probability
                else:
                    action_cumsum = np.cumsum(masked_output)
                    action = (action_cumsum >
                              np.random.randint(1, pm.RAND_RANGE) /
                              float(pm.RAND_RANGE)).argmax()

            action_vec[action] = 1
            # check whether skip this timeslot
            if pm.SKIP_TS and action == len(action_vec) - 1:
                self._move()
                # filter out the first action that causes 0 reward??? NO
                # if sum([job.num_workers+job.num_ps for job in self.uncompleted_jobs]) > 0:
                valid_state = True
                self.sched_seq.append(None)
                self.logger.debug("Skip action is selected!")
                self.logger.debug("Output: " + str(output))
                self.logger.debug("Masked output: " + str(masked_output))
            else:
                # count action freq
                if pm.PS_WORKER and pm.BUNDLE_ACTION:
                    self.action_freq[action % 3] += 1

                # allocate resource
                if pm.PS_WORKER:
                    if pm.BUNDLE_ACTION:
                        job = self.window_jobs[action / 3]
                    else:
                        job = self.window_jobs[action / 2]
                else:
                    job = self.window_jobs[action]
                if job is None:
                    self._move()
                    self.logger.debug("The selected action is None!")
                else:
                    _, node = self.node_used_resr_queue.get()
                    # get resource requirement of the selected action
                    if pm.PS_WORKER:
                        if pm.BUNDLE_ACTION:
                            if action % 3 == 0:
                                resr_reqs = job.resr_worker
                            elif action % 3 == 1:
                                resr_reqs = job.resr_ps
                            else:
                                resr_reqs = job.resr_worker + job.resr_ps
                        else:
                            if action % 2 == 0:  # worker
                                resr_reqs = job.resr_worker
                            else:
                                resr_reqs = job.resr_ps
                    else:
                        resr_reqs = job.resr_worker
                    succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)
                    if succ:
                        move_on = False
                        # change job tasks and placement
                        if pm.PS_WORKER:
                            if pm.BUNDLE_ACTION:
                                if action % 3 == 0:  # worker
                                    job.num_workers += 1
                                    job.curr_worker_placement.append(node)
                                elif action % 3 == 1:  # ps
                                    job.num_ps += 1
                                    job.curr_ps_placement.append(node)
                                else:  # bundle
                                    job.num_ps += 1
                                    job.curr_ps_placement.append(node)
                                    job.num_workers += 1
                                    job.curr_worker_placement.append(node)
                            else:
                                if action % 2 == 0:  # worker
                                    job.num_workers += 1
                                    job.curr_worker_placement.append(node)
                                else:  # ps
                                    job.num_ps += 1
                                    job.curr_ps_placement.append(node)
                        else:
                            job.num_workers += 1
                            job.curr_worker_placement.append(node)

                        job.dom_share = np.max(
                            1.0 * (job.num_workers * job.resr_worker +
                                   job.num_ps * job.resr_ps) /
                            self.cluster.CLUSTER_RESR_CAPS)
                        self.node_used_resr_queue.put(
                            (np.sum(node_used_resrs), node))
                        self.running_jobs.add(job)
                        valid_state = True
                        self.sched_seq.append(job)
                    else:
                        self._move()
                        self.logger.debug("No enough resources!")
        if move_on:
            reward = self.rewards[-1] * move_on
        else:
            reward = 0
        return masked_output, action_vec, reward, move_on, valid_state  # invalid state, action and output when move on except for skip ts

    def get_jobstats(self):
        self.jobstats["duration"] = [(job.end_time - job.arrv_time + 1)
                                     for job in self.completed_jobs]
        for name, value in self.jobstats.items():
            self.logger.debug(name + ": length " + str(len(value)) + " " +
                              str(value))
        return self.jobstats

    def _sched_states(self):
        self.states = []
        for job in self.running_jobs:
            self.states.append((job.id, job.type, job.num_workers, job.num_ps))

    def get_job_reward(self):
        job_reward = []
        for job in self.sched_seq:
            if job is None:  # skip
                if len(self.job_prog_in_ts) > 0:
                    job_reward.append(self.rewards[-1] /
                                      len(self.job_prog_in_ts))
                else:
                    job_reward.append(0)
            else:
                job_reward.append(self.job_prog_in_ts[job])
        self.sched_seq = []
        self.job_prog_in_ts.clear()

        self.logger.info("Action Frequency: " + str(self.action_freq))
        return job_reward

    def get_sched_states(self):
        return self.states

    def _progress(self):
        reward = 0
        num_ts_completed = 0
        for job in self.running_jobs:
            norm_prog = job.step() / job.num_epochs
            self.job_prog_in_ts[job] = norm_prog
            reward += norm_prog
            if job.progress >= job.real_num_epochs:
                if pm.FINE_GRAIN_JCT:
                    job.end_time = self.curr_ts - 1 + job.get_run_time_in_ts()
                else:
                    job.end_time = self.curr_ts
                # self.running_jobs.remove(job) # it means running in this ts, so no need to delete
                self.uncompleted_jobs.remove(job)
                self.completed_jobs.add(job)
                num_ts_completed += 1
        self.rewards.append(reward)

        self.jobstats["running"].append(len(self.running_jobs))
        self.jobstats["tot_completed"].append(len(self.completed_jobs))
        self.jobstats["uncompleted"].append(len(self.uncompleted_jobs))
        self.jobstats["ts_completed"].append(num_ts_completed)
        cpu_util, gpu_util = self.cluster.get_cluster_util()
        self.jobstats["cpu_util"].append(cpu_util)
        self.jobstats["gpu_util"].append(gpu_util)


def test():
    import log, trace
    logger = log.getLogger(name="agent_" + str(id), level="INFO")
    job_trace = trace.Trace(logger).get_trace()
    env = RL_Env("RL", job_trace, logger)
    while not env.end:
        data = env.step()
        for item in data:
            print item
        print "-----------------------------"
        raw_input("Next? ")

    print env.get_results()


if __name__ == '__main__':
    test()
