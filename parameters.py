# experiment use
EXPERIMENT_NAME = None

# random seed
RANDOMNESS = False
np_seed = 9973  # seed for numpy
tf_seed = 53  # seed for tf
trace_seed = 103  # seed for trace, not used

# configuration
LOG_MODE = "INFO"
if LOG_MODE == "DEBUG":
	NUM_AGENTS = 1
else:
	NUM_AGENTS = 1  # at most 28 for tesla p100 and 40 for gtx 1080ti

TRAINING_MODE = "RL"  # or "RL"
if TRAINING_MODE == "SL":
	HEURISTIC = "DRF"  # the heuristic algorithm used for supervised learning
if TRAINING_MODE == "RL":
	VALUE_NET = True  # disable/enable critic network
else:
	VALUE_NET = False

POLICY_NN_MODEL = None #"Models/policy_sl_ps_worker_100.ckpt"  # path of the checkpointed model, or None
VALUE_NN_MODEL = None  # "Models/value_rl_ps_worker_1000.ckpt"  # path of value network model
SAVE_VALUE_MODEL = True
if TRAINING_MODE == "SL" or VALUE_NN_MODEL is not None:
	SAVE_VALUE_MODEL = False
if TRAINING_MODE == "SL":
	POLICY_NN_MODEL = None
	VALUE_NN_MODEL = None
SUMMARY_DIR = "TensorBoard/"  # tensorboard logging dir
MODEL_DIR = "Models/"  # checkpoint dir
MAX_NUM_CHECKPOINTS = 10  # max number of saved checkpoints
CHECKPOINT_INTERVAL = 10000
if TRAINING_MODE == "SL":
	CHECKPOINT_INTERVAL = 100
DISP_INTERVAL = 50  # display frequency
VISUAL_GW_INTERVAL = 1000  # tf log gradients/weights frequency
NUM_RECORD_AGENTS = 2  # log details of 2 agents in tensorboard and ignore others for saved space
SKIP_FIRST_VAL = False  # if False, the central agent will test the initialized model at first before training
SELECT_ACTION_MAX_PROB = False  # whether to select the action with the highest probability or select based on distribution, default based on distribution
MASK_PROB = 1.  # whether to mask actions mapped None jobs, set it to be lower seems to be worse
ASSIGN_BUNDLE = True  # assign 1 ps and 1 worker for each in the beginning of each timeslot to avoid starvation

# hyperparameters
SL_LOSS_FUNCTION = "Cross_Entropy"  # "Mean_Square", "Cross_Entropy", "Absolute_Difference"
OPTIMIZER = "Adam"  # RMSProp
FIX_LEARNING_RATE = True  # keep constant learning rate
ADJUST_LR_STEPS = [5000]  # halving learning rate once reaching a certain step, not functional
if TRAINING_MODE == "SL":
	LEARNING_RATE = 0.005
else:
	LEARNING_RATE = 0.0001

MINI_BATCH_SIZE = 256/NUM_AGENTS
EPSILON_GREEDY = False  # whether to enable epsilon greedy policy for exploration
VARYING_EPSILON = True  # different values of epsilon for agents
EPSILON = 0.1  # not used
ENTROPY_WEIGHT = 0.1
ENTROPY_EPS = 1e-6
MAX_ENTROPY_WEIGHT = 10.0
ANNEALING_TEMPERATURE = 500.0
FIX_ENTROPY_WEIGHT = True  # if true, the entropy weight is ENTROPY_WEIGHT; else, it is calculated based on ANNEALING_TEMPERATURE and MAX_ENTROPY_WEIGHT
if not FIX_ENTROPY_WEIGHT:
	assert not EPSILON_GREEDY  # avoid using varying entropy and e-greedy together

RAND_RANGE = 100000
TOT_TRAIN_EPOCHS = 2000  # number of training epochs
TOT_NUM_STEPS = 1000000
if TRAINING_MODE == "SL":
	TOT_NUM_STEPS = 100000
VAL_INTERVAL = 200  # validation interval
if TRAINING_MODE == "SL":
	VAL_INTERVAL = 50
VAL_ON_MASTER = True  # validation on agent uses CPU instead of GPU, and may cause use up all memory, do not know why, so far it must be set true

REPLAY_MEMORY_SIZE = 8192  # or 65536
RANDOM_FILL_MEMORY = False
PRIORITY_REPLAY = True
PRIORITY_MEMORY_SORT_REWARD = True  # use reward as priority
PRIORITY_MEMORY_EVICT_PRIORITY = False  # remove samples from experience buffer based on priority instead of age
PRIORITY_MEMORY_SORT_WEIGHTED_REWARD_GRADIENTS = False  # not used
assert PRIORITY_MEMORY_SORT_REWARD + PRIORITY_MEMORY_SORT_WEIGHTED_REWARD_GRADIENTS <= 1
if TRAINING_MODE == "SL":
	PRIORITY_REPLAY = False

LT_REWARD_IN_TS = False  # use long term reward within a timeslot
LT_REWARD_NUM_TS = 1  # not implemented
DISCOUNT_FACTOR = 0.99
TS_REWARD_PLUS_JOB_REWARD = False  # another way to assign reward
NUM_UNCOMPLETED_JOB_REWARD = False  # set the reward to be the number of uncompleted jobs
assert LT_REWARD_IN_TS+TS_REWARD_PLUS_JOB_REWARD <= 1
MEAN_REWARD_BASELINE = True  # whether to use reward mean as baseline
if MEAN_REWARD_BASELINE:
	assert PRIORITY_MEMORY_SORT_REWARD

INJECT_SAMPLES = True  # inject samples to experience buffer to get samples with high reward
SAMPLE_INJECTION_PROB = 0.1  # probabilistically inject samples with high reward
VARYING_SKIP_NUM_WORKERS = True
MIN_ACTION_PROB_FOR_SKIP = 10**(-20)  # 10**(-12)
NUM_TS_PER_UPDATE = 1  # update once after passing x timeslot(s), default 1, i.e., update weights per timeslot
VARYING_PS_WORKER_RATIO = True  # explore different ratio of ps over worker
STEP_TRAIN_CRITIC_NET = 0  # number of steps for pretraining critic network, default 0, not functional
CHANGING_JOB_TYPES = False
JOB_RESR_BALANCE = True
FINE_GRAIN_JCT = True

# cluster
TESTBED = False
LARGE_SCALE = True
assert TESTBED+LARGE_SCALE < 2
CLUSTER_NUM_NODES = 96  # should be at least 3 times of maximal number of uncompleted jobs at each ts, default 160
if TESTBED:
	CLUSTER_NUM_NODES = 6
elif LARGE_SCALE:
	CLUSTER_NUM_NODES = 500
NUM_RESR_TYPES = 2  # number of resource types, e.g., cpu,gpu
NUM_RESR_SLOTS = 8  # number of available resource slots on each machine

# dataset
JOB_EPOCH_EST_ERROR = 0
TRAIN_SPEED_ERROR = 0
REAL_SPEED_TRACE = True  # whether to use real traces collected from experiment testbed
FIX_JOB_LEN = True
JOB_LEN_PATTERN = "Ali_Trace"  # Ali_Trace, Normal
JOB_ARRIVAL_PATTERN = "Ali_Trace"  # Ali_Trace, Uniform, Google_Trace, Poisson
TRAIN_EPOCH_SIZE = 100  # number of traces for training dataset
TOT_NUM_JOBS = 60  # number of jobs in one trace
MAX_ARRVS_PER_TS = 3  # max number of jobs arrived in one time slot
MAX_NUM_EPOCHS = 30000   # maximum duration of jobs, epochs. default 200
MAX_NUM_WORKERS = 16
TS_DURATION = 1200.0
if LARGE_SCALE:
	TOT_NUM_JOBS = 200  # number of jobs in one trace
	MAX_ARRVS_PER_TS = 10  # max number of jobs arrived in one time slot
if TESTBED:
	TOT_NUM_JOBS = 10
	MAX_NUM_EPOCHS = 1000
	MAX_ARRVS_PER_TS = 5
	TS_DURATION = 300.0
	SCHED_WINDOW_SIZE = 4
VAL_DATASET = 10  # number of traces for validation in each agent
MAX_TS_LEN = 1000  # maximal timeslot length for one trace


# neural network
JOB_ORDER_SHUFFLE = False  # whether to shuffle the order of the jobs in the scheduling window, can also be used for data augmentation
if TRAINING_MODE == "SL":
	JOB_ORDER_SHUFFLE = True
JOB_SORT_PRIORITY = "Arrival" # or Arrival, Resource, Progress, sort job based on resource or arrival
SCHED_WINDOW_SIZE = 20  # maximum allowed number of jobs for NN input
if LARGE_SCALE:
	SCHED_WINDOW_SIZE = 40
PS_WORKER = True  # whether consider ps and worker tasks separately or not
INPUTS_GATE=[("TYPE",True), ("STAY",True), ("PROGRESS",True), ("DOM_RESR",True), ("WORKERS",True)]
if PS_WORKER:
	INPUTS_GATE.append(("PS", True))
	INJECT_SAMPLES = True
BUNDLE_ACTION = True  # add a 'bundle' action to each job, i.e., selecting a ps and a worker by one action
TYPE_BINARY = False  # 4 bits
type_str, enable = INPUTS_GATE[0]
if not enable:
	TYPE_BINARY = False
STATE_DIM = (3*TYPE_BINARY + sum([enable for (_,enable) in INPUTS_GATE]), SCHED_WINDOW_SIZE)  # type, # of time slots in the system so far, normalized remaining epoch, dom resource, # of workers
SKIP_TS = True    # whether we skip the timeslot
if PS_WORKER:
	ACTION_DIM = 2 * SCHED_WINDOW_SIZE + SKIP_TS
	if BUNDLE_ACTION:
		ACTION_DIM = 3 * SCHED_WINDOW_SIZE + SKIP_TS
else:
	ACTION_DIM = SCHED_WINDOW_SIZE + SKIP_TS

INPUT_RESCALE = False  # not implemented on heuristic algorithms yet
JOB_CENTRAL_REPRESENTATION = False  # treat each job as an input instead of treating each type of information of all jobs as an input
ATTRIBUTE_CENTRAL_REPRESENTATION = False  # treat each property of all jobs as an input, default fully connected to input
ZERO_PADDING = True  # how to represent None job as input
FIRST_LAYER_TANH = False
NN_SHORTCUT_CONN = False  # connect the output of first layer to the NN layer before softmax output
if NN_SHORTCUT_CONN:
	assert JOB_CENTRAL_REPRESENTATION  # must enable JOB_CENTRAL_REPRESENTATION
NUM_FCN_LAYERS = 2  # number of fully connected layers, must be > 0
NUM_NEURONS_PER_FCN = STATE_DIM[0] * STATE_DIM[1] * 2 / 3  # default same number as input size
BATCH_NORMALIZATION = True

