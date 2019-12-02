import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.embedding_nn_normalized_v2 as nn_model
from fast_adaptation_embedding.models.ffnn import FFNN_Ensemble_Model
from fast_adaptation_embedding.controllers.cem import CEM_opt
from fast_adaptation_embedding.controllers.random_shooting import RS_opt
import utils
import torch
import numpy as np
import copy
import gym
import time
from datetime import datetime
import pickle
import os
from os import path
# from pyprind import ProgBar
from utils import ProgBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import argparse
from tqdm import tqdm
import math

import rospy
import ite_minitaur_utils as robot_utils

def frac(x):
	return x - np.floor(x)


def sawtooth(t, freq, phase=0):
	T = 1/float(freq)
	y = frac(t/T + phase)
	return y

def controller_sawtooth(params, t):
	"""Sawtooth controller"""
	# ~ steer = params[0] if self._unblocked_steering else 0  # Move in different directions
	steer = 0
	step_size = params[0] * 0.5 + 0.5  # Walk with different step_size forward or backward
	leg_extension = params[1]  # Walk on different terrain
	leg_extension_offset = params[2]
	swing_offset = params[3]  # Walk in slopes

	# Robot specific parameters
	swing_limit = 0.6
	extension_limit = 1
	speed = 2.0  # cycle per second
	swing_offset = swing_offset * 0.2

	extension = extension_limit * (leg_extension + 1) * 0.5
	leg_extension_offset = 0.1 * leg_extension_offset - 0.8
	# Steer modulates only the magnitude (abs) of step_size
	A = np.clip(abs(step_size) + steer, 0, 1) if step_size >= 0 else -np.clip(abs(step_size) + steer, 0, 1)
	B = np.clip(abs(step_size) - steer, 0, 1) if step_size >= 0 else -np.clip(abs(step_size) - steer, 0, 1)

	# We want legs to move sinusoidally, smoothly
	fl = math.sin(t * speed * 2 * np.pi) * (swing_limit * A) + swing_offset
	br = math.sin(t * speed * 2 * np.pi) * (swing_limit * B) + swing_offset
	fr = math.sin(t * speed * 2 * np.pi + math.pi) * (swing_limit * B) + swing_offset
	bl = math.sin(t * speed * 2 * np.pi + math.pi) * (swing_limit * A) + swing_offset

	# Sawtooth for faster contraction
	e1 = extension * sawtooth(t, speed, 0.25) + leg_extension_offset
	e2 = extension * sawtooth(t, speed, 0.5 + 0.25) + leg_extension_offset
	return np.clip(np.array([fl, bl, fr, br, -e1, -e2, -e2, -e1]), -1, 1)

def ConvertFromLegModel(actions):
	"""Convert the actions that use leg model to the real motor actions.

	Args:
	  actions: [sequence_length, swing+extention] where swing, extention = [LF, LB, RF, RB]
	Returns:
	  The eight desired motor angles that can be used in ApplyActions().
	  the orientation of theta1 and theta2 are opposite
	  [sequence_length, LF+LB+RF+RB] where LF=LB=RF=RB=(theta1, theta2)
	"""
	motor_angle = np.copy(actions)
	scale_for_singularity = 1
	offset_for_singularity = 1.5
	half_num_motors = 4
	quater_pi = math.pi / 4
	for i in range(8):
		action_idx = int(i // 2)
		forward_backward_component = (
				-scale_for_singularity * quater_pi *
				(actions[:, action_idx + half_num_motors] + offset_for_singularity))
		extension_component = (-1) ** i * quater_pi * actions[:, action_idx]
		if i >= half_num_motors:
			extension_component = -extension_component
		motor_angle[:, i] = (math.pi + forward_backward_component + extension_component)
	return motor_angle

def polar(motor_angles):
	"""
	Args:
	  The eight desired motor angles
	  [sequence_length, LF+LB+RF+RB] where LF=LB=RF=RB=(theta1, theta2)
	Returns:
		r: array of radius for each leg [sequence_length, 4]
		theta: array of polar angle [sequence_length, 4]
		(legs order [LF, LB, RF, RB])
	"""
	l1 = 0.1
	l2 = 0.2
	R = []
	Theta = []
	for i in range(4):
		o1, o2 = motor_angles[:, 2 * i], motor_angles[:, 2 * i + 1]
		beta = (o1 + o2) / 2
		r = np.sqrt(l2 ** 2 - (l1 * np.sin(beta)) ** 2) - l1 * np.cos(beta)
		R.append(r)
		Theta.append(np.pi + (o1 - o2) / 2)
	return R, Theta


def cartesian(r, theta):
	"""
	Args:
		r: array of radius for each leg [sequence_length, 4]
		theta: array of polar angle [sequence_length, 4]
		(legs order [LF, LB, RF, RB])

	Returns:
		(x,y) array of cartesian position of the foot of each legs x=y=[sequence_length, 4]
	"""
	return (r * np.cos(theta), r * np.sin(theta))

def paramToCartesianTrajectory(param, control_step, K):
	actions = np.array([controller_sawtooth(param, t*control_step) for t in range(K)])
	m_action = ConvertFromLegModel(actions)
	R, Theta = polar(np.array(m_action))
	(x, y) = cartesian(R, Theta)
	return (x,y)

class Cost(object):
	def __init__(self, model, init_state, horizon, action_dim, goal, pred_high, pred_low, config, task_likelihoods):
		self.__init_state = init_state
		self.__horizon = horizon
		self.__action_dim = action_dim
		self.__goal = goal
		self.__models = model
		self.__pred_high = pred_high
		self.__pred_low = pred_low
		self.__obs_dim = len(init_state)
		self.__discount = config['discount']
		self.__batch_size = config['ensemble_batch_size']
		self.__xreward = config['xreward']
		self.__yreward = config['yreward']

	def cost_fn(self, samples):
		action_samples = torch.FloatTensor(samples).cuda() \
			if self.__models[0].cuda_enabled  \
			else torch.FloatTensor(samples)
		init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda() \
			if self.__models[0].cuda_enabled  \
			else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
		all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() \
			if self.__models[0].cuda_enabled  \
			else torch.FloatTensor(np.zeros(len(samples)))

		n_batch = max(1, int(len(samples) / self.__batch_size))
		per_batch = len(samples) / n_batch

		for i in range(n_batch):
			start_index = int(i * per_batch)
			end_index = len(samples) if i == n_batch - 1 else int(i * per_batch + per_batch)
			action_batch = action_samples[start_index:end_index]
			start_states = init_states[start_index:end_index]
			dyn_model = self.__models[np.random.randint(0, len(self.__models))]
			for h in range(self.__horizon):
				actions = action_batch[:, h * self.__action_dim: h * self.__action_dim + self.__action_dim]
				model_input = torch.cat((start_states, actions), dim=1)
				diff_state = dyn_model.predict_tensor(model_input)
				start_states += diff_state[:, 2:]
				x_vel_cost = -diff_state[:, 0] * self.__xreward
				y_vel_cost = -diff_state[:, 1] * self.__yreward
				all_costs[start_index: end_index] += x_vel_cost * self.__discount ** h + y_vel_cost * self.__discount ** h
		return all_costs.cpu().detach().numpy()

def train_meta(tasks_in, tasks_out, config):
	model = nn_model.Embedding_NN(dim_in=config["dim_in"],
								hidden=config["hidden_layers"],
								dim_out=config["dim_out"],
								embedding_dim=config["embedding_size"],
								num_tasks=len(tasks_in),
								CUDA=config["cuda"],
								SEED=None,
								output_limit=config["output_limit"],
								dropout=0.0,
								hidden_activation=config["hidden_activation"])
	task_losses = nn_model.train_meta(model,
						tasks_in,
						tasks_out,
						meta_iter=config["meta_iter"],
						inner_iter=config["inner_iter"],
						inner_step=config["inner_step"],
						meta_step=config["meta_step"],
						minibatch=config["meta_batch_size"],
						inner_sample_size=config["inner_sample_size"])
	return  model, task_losses

def train_model(model, train_in, train_out, task_id, config):
	cloned_model =copy.deepcopy(model)
	nn_model.train(cloned_model,
				train_in,
				train_out,
				task_id=task_id,
				inner_iter=config["epoch"],
				inner_lr=config["learning_rate"],
				minibatch=config["minibatch_size"])
	return cloned_model

def train_ensemble_model(train_in, train_out, sampling_size, config, model= None):
	network = model
	if network is None:
		network = FFNN_Ensemble_Model(dim_in=config["ensemble_dim_in"],
									hidden=config["ensemble_hidden"],
									hidden_activation=config["hidden_activation"],
									dim_out=config["ensemble_dim_out"],
									CUDA=config["ensemble_cuda"],
									SEED=config["ensemble_seed"],
									output_limit=config["ensemble_output_limit"],
									dropout=config["ensemble_dropout"],
									n_ensembles=config["n_ensembles"])

	network.train(epochs=config["ensemble_epoch"],
				training_inputs=train_in,
				training_targets=train_out,
				batch_size=config["ensemble_batch_size"],
				logInterval=config["ensemble_log_interval"],
				sampling_size=sampling_size)
	return copy.deepcopy(network)

def process_data(data):
	'''Assuming dada: an array containing [state, action, state_transition, cost] '''
	training_in = []
	training_out = []
	for d in data:
		s = d[0]
		a = d[1]
		training_in.append(np.concatenate((s,a)))
		training_out.append(d[2])
	return np.array(training_in), np.array(training_out), np.max(training_in, axis=0), np.min(training_in, axis=0)

def execute(env, init_state, steps, init_mean, init_var, model, config, last_action_seq, task_likelihoods, pred_high, pred_low, recorder, current_state):
	# current_state = env.reset()
	robot_utils.end_episode_distance = -1000
	prev_state = -1000
	# current_state = copy.copy(env.state) if config['online'] else env.reset()
	try:
		config["goal"]=env.goal
	except:
		pass
	trajectory = []
	traject_cost = 0
	sliding_mean = init_mean #np.zeros(config["sol_dim"])

	temp_config = copy.deepcopy(config)
	temp_config["popsize"] = 20000
	optimizer = None
	sol = None
	for i in tqdm(range(steps)):
		print("current_state ", current_state)
		cost_object = Cost(model=model, init_state=current_state[2:], horizon=config["horizon"],
						task_likelihoods=task_likelihoods, action_dim=env.action_space.shape[0], goal=config["goal"],
						pred_high=pred_high, pred_low=pred_low, config=config)
		config["cost_fn"] = cost_object.cost_fn
		optimizer = RS_opt(config)
		# sol = optimizer.obtain_solution(sliding_mean, init_var)
		sol = optimizer.obtain_solution()

		a = sol[0:env.action_space.shape[0]]


		ctrl_to_test = a
		print("ctrl_to_test: ", ctrl_to_test)
		input("Press enter to send ctrl ... ")
		'''------------------------Test the ctrl on the real robot------------------------------------'''
		robot_utils.eval_ros(episode_duration,ctrl_to_test,rate,1)
		print("state " , robot_utils.state)
		next_state = np.array(robot_utils.state)


		# if(robot_utils.end_episode_distance == prev_ep_distance):
		# 	sys.exit("WARNING : ite hasn't been able to recover the latest distance, exiting ite")
		# else:
		# 	prev_ep_distance = robot_utils.end_episode_distance
		# 	real_perf = robot_utils.end_episode_distance
		# print("real_perf: ", ctrl_to_test)
		# input("Press enter to continue... ")


		# ~ paramToCartesianTrajectory(a,0.004,125)
		# next_state, r = 0, 0
		# for k in range(config['K']):
		# 	if config["record_video"]:
		# 		recorder.capture_frame()
		# 	next_state, rew, _, _ = env.step(a)
		# 	r += rew

		r = next_state[0] #x travelled

		trajectory.append([current_state[2:].copy(), a.copy(), next_state-current_state, -r])
		current_state = next_state
		traject_cost += -r

	if config["record_video"]:
		recorder.capture_frame()
		recorder.close()
	return trajectory, traject_cost, current_state

def test_model(ensemble_model, init_state, action, state_diff):
	x = np.concatenate(([init_state], [action]), axis=1)
	y = state_diff.reshape(1,-1)
	y_pred = ensemble_model.get_models()[0].predict(x)
	# print("True: ", y.flatten())
	# print("pred: ", y_pred.flatten())
	# input()
	return np.power(y-y_pred,2).sum()

def extract_action_seq(data):
	actions = []
	for d in data:
		actions += d[1].tolist()
	return np.array(actions)

def compute_likelihood(data, models, adapt_steps, beta=1.0):
	'''
	Computes MSE loss and then softmax to have a probability
	'''
	data_size = config['adapt_steps']
	if data_size is None: data_size = len(data)
	lik = np.zeros(len(models))
	x, y, _, _ = process_data(data[-data_size::])
	for i,m in enumerate(models):
		y_pred = m.predict(x)
		lik[i] = np.exp(- beta * m.loss_function_numpy(y, y_pred)/len(x))
	return lik/np.sum(lik)

def sample_model_index(likelihoods):
	cum_sum = np.cumsum(likelihoods)
	num = np.random.rand()
	for i, cum_prob in enumerate(cum_sum):
		if num <= cum_prob: return i




def main(gym_args, config, test_mismatches, rate, episode_duration, gym_kwargs={}):

	'''---------Prepare the directories------------------'''
	if config['logdir'] is None:
		now = datetime.now()
		timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
		experiment_name = timestamp + "_" + config["exp_suffix"]
		res_dir = os.path.join(os.getcwd(), config["result_dir"], config["env_name"],experiment_name)
		try:
			i = 0
			while True:
				res_dir += "_" + str(i)
				i += 1
				if not os.path.isdir(res_dir):
					os.makedirs(res_dir)
					os.makedirs(res_dir+"/videos")
					break
		except:
			print("Could not make the result directory!!!")
	else:
		res_dir = config['logdir']
		os.makedirs(res_dir+"/videos")


	with open(res_dir + "/details.txt","w+") as f:
		f.write(config["exp_details"])

	with open(res_dir + '/config.json', 'w') as fp:
		import json
		json.dump(config, fp)


	'''---------Prepare the test environment---------------'''
	env = gym.make(*gym_args, **gym_kwargs)
	trained_mismatches = np.load(config["data_dir"] + "/mismatches.npy")
	n_training_tasks = len(trained_mismatches)
	try:
		s = os.environ['DISPLAY']
		print("Display available")
		# env.render(mode="rgb_array")
		env.render(mode="human")
		env.reset()
	except:
		print("Display not available")
		env.reset()

	print("\n\n\n")
	'''---------Initialize global variables------------------'''
	data = []
	models = []
	best_action_seq = np.random.rand(config["sol_dim"])*2.0 - 1.0
	best_cost = 10000
	last_action_seq = None
	all_action_seq = []
	all_costs = []
	with open(res_dir + "/costs.txt","w+") as f:
		f.write("")

	'''--------------------Meta learn the models---------------------------'''
	meta_model = None
	if not path.exists(config["data_dir"] + "/"+ config["model_name"]+".pt"):
		print("Model not found. Learning from data...")
		meta_data = np.load(config["data_dir"] + "/trajectories.npy", allow_pickle=True)
		tasks_in, tasks_out = [], []
		for n in range(n_training_tasks):
			x, y, high, low = process_data(meta_data[n])
			tasks_in.append(x)
			tasks_out.append(y)
			print("task ", n, " data: ", len(tasks_in[n]), len(tasks_out[n]))
		meta_model, task_losses = train_meta(tasks_in, tasks_out, config)
		meta_model.save(config["data_dir"] + "/"+ config["model_name"]+".pt")
		np.save(config["data_dir"] + "/"+config["model_name"]+"_task_losses.npy", task_losses)
	else:
		print("Model found. Loading from '.pt' file...")
		device =  torch.device("cuda") if config["cuda"] else torch.device("cpu")
		meta_model = nn_model.load_model(config["data_dir"] + "/"+ config["model_name"]+".pt", device)

	raw_models =  [copy.deepcopy(meta_model) for _ in range(n_training_tasks)]
	models =  [copy.deepcopy(meta_model) for _ in range(n_training_tasks)]
	for task_id,m in enumerate(raw_models):
		m.fix_task(task_id)

	for task_id,m in enumerate(models):
		m.fix_task(task_id)

	input("Init Ok press enter to start testing ... ")
	'''------------------------Test time------------------------------------'''
	n_test_tasks = len(test_mismatches)
	Data = []
	Likelihoods = []
	current_state = np.array([0,0,0,1])
	for mismatch_index in range(n_test_tasks):
		print('Task :', mismatch_index)
		high, low =  np.ones(config["dim_out"])*1000.,  -np.ones(config["dim_out"])*1000.
		task_likelihoods = np.random.rand(n_training_tasks)
		data = []
		likelihoods = []
		for index_iter in range(config["iterations"]):
			print("Episode: ", index_iter)
			new_mismatch = test_mismatches[mismatch_index]
			print("Mismatch: ", new_mismatch.tolist())
			env.set_mismatch(new_mismatch)
			recorder = VideoRecorder(env, res_dir + "/videos/" + str(index_iter) +".mp4") if config["record_video"] else None
			trajectory, c, current_state = execute(env=env,
									init_state=config["init_state"],
									model=models,
									steps= config["episode_length"],
									init_mean=np.zeros(config["sol_dim"]) ,
									init_var=0.01 * np.ones(config["sol_dim"]),
									config=config,
									last_action_seq = None,
									task_likelihoods=task_likelihoods,
									pred_high= high,
									pred_low = low,
									recorder=recorder,
									current_state = current_state)

			data += trajectory
			'''-----------------Compute likelihood before relearning the models-------'''
			task_likelihoods = compute_likelihood(data, raw_models, config['adapt_steps'])
			print("\nlikelihoods: ", task_likelihoods)
			likelihoods.append(task_likelihoods)
			x, y, high, low = process_data(data)

			task_index = sample_model_index(task_likelihoods) if config["sample_model"] else np.argmax(task_likelihoods)
			print("\nEstimated task-id: ", task_index)
			task_likelihoods = task_likelihoods * 0
			task_likelihoods [task_index] = 1.0
			data_size = config['adapt_steps']
			if data_size is None: data_size = len(x)
			print("Learning model with recent ", data_size, " data")
			models[task_index] = train_model(model=copy.deepcopy(raw_models[task_index]), train_in=x[-data_size::], train_out=y[-data_size::], task_id=task_index, config=config)

			print("\nCost : ", c)
			with open(res_dir +"/costs.txt","a+") as f:
				f.write(str(c)+"\n")

			if c<best_cost:
				best_cost = c
				best_action_seq = []
				for d in trajectory:
					best_action_seq += d[1].tolist()
				best_action_seq = np.array(best_action_seq)
				last_action_seq = extract_action_seq(trajectory)

			all_action_seq.append(extract_action_seq(trajectory))
			all_costs.append(c)
		Data.append(data)
		Likelihoods.append(likelihoods)
	np.save(res_dir + "/trajectories.npy", Data)
	np.save(res_dir + "/test_mismatches.npy", test_mismatches)
	np.save(res_dir + "/likelihoods.npy", Likelihoods)
	print("\n********************************************************\n")

#######################################################################################################

config = {
	#exp parameters:
	"horizon": 1, #NOTE: "sol_dim" must be adjusted
	"iterations": 10,
	"random_episodes": 1, # per task
	"episode_length": 5, #number of times the controller is updated
	"online": False,
	"adapt_steps": None,
	"init_state": None, #Must be updated before passing config as param
	"action_dim":4,
	"goal": None, #Sampled durng env reset
	"ctrl_time_step": 0.02,
	"K": 50, #number of control steps with the same control1ler
	"xreward": 1,
	"yreward": 0,
	"record_video": False,
	"online_damage_probability": 0.0,
	"sample_model": False,

	#logging
	"result_dir": "results",
	"data_dir": "data/test/",
	"model_name": "controlled_minitaur_meta_embedding_model",
	"env_name": "meta_controlled_minitaur",
	"exp_suffix": "experiment",
	"exp_details": "Default experiment.",
	"logdir": None,

	#Model_parameters
	"dim_in": 4+2,
	"dim_out": 4,
	"hidden_layers": [20, 20],
	"embedding_size": 8,
	"cuda": False,
	"output_limit": 10.0,
	"ensemble_batch_size": 16384,

	#Meta learning parameters
	"meta_iter": 200,#5000,
	"meta_step": 0.3,
	"inner_iter": 10,#10,
	"inner_step": 0.0001,
	"meta_batch_size": 32,
	"inner_sample_size":500,

	#Model learning parameters
	"epoch": 20,
	"learning_rate": 1e-4,
	"minibatch_size": 32,
	"hidden_activation": "relu",

	#Optimizer parameters
	"max_iters": 5,
	"epsilon": 0.0001,
	"lb": -1.,
	"ub": 1.,
	"popsize": 2000,
	"sol_dim": 4*10, #NOTE: Depends on Horizon
	"num_elites": 30,
	"cost_fn": None,
	"alpha": 0.1,
	"discount":1.0
}

#optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations",
					help='Total episodes in episodic learning. Total MPC steps in the experiment.',
					type=int)
parser.add_argument("--data_dir",
					help='Path to load dynamics data and/or model',
					type=str)
parser.add_argument("--exp_details",
					help='Details about the experiment',
					type=str)
parser.add_argument("--online",
					action='store_true',
					help='Will not reset back to init position', )
parser.add_argument("--adapt_steps",
					help='Past steps to be used to learn a new model from the meta model',
					type=int)
parser.add_argument("--control_steps",
					help='Steps after which learn a new model => Learning frequency.',
					type=int)
parser.add_argument("--rand_motor_damage",
					action='store_true',
					help='Sample a random joint damage.')
parser.add_argument("--rand_orientation_fault",
					action='store_true',
					help='Sample a random orientation estimation fault.')
parser.add_argument("--sample_model",
					action='store_true',
					help='Sample a model (task-id) using the likelihood information. Default: Picks the most likely model.')
parser.add_argument("--online_damage_probability",
					help='Sample probabilistically random mismatch during mission. NOT used for episodic testing',
					default = 0.0,
					type=float)
parser.add_argument('-c', '--config', action='append', nargs=2, default=[])
parser.add_argument('-logdir', type=str, default='None')

arguments = parser.parse_args()
if arguments.data_dir is not None: config['data_dir'] = arguments.data_dir
if arguments.iterations is not None: config['iterations'] = arguments.iterations
if arguments.exp_details is not None: config['exp_details'] = arguments.exp_details
if arguments.online is True:
	config['online'] = True
	if arguments.adapt_steps is not None: config['adapt_steps'] = arguments.adapt_steps
	if arguments.control_steps is not None: config['episode_length'] = arguments.control_steps
	if arguments.online_damage_probability is not None: config['online_damage_probability'] = arguments.online_damage_probability
	print("Online learning with adaptation steps: ", config['adapt_steps'], " control steps: ", config['episode_length'])
else:
	print("Episodic learning with episode length: ", config['episode_length'])

if arguments.rand_motor_damage is not None: config['rand_motor_damage'] = arguments.rand_motor_damage
if arguments.rand_orientation_fault is not None: config['rand_orientation_fault'] = arguments.rand_orientation_fault
if arguments.sample_model is not None: config['sample_model'] = arguments.sample_model


logdir = arguments.logdir
config['logdir'] = None if logdir == 'None' else logdir

for (key, val) in arguments.config:
	if key in ['horizon', 'K', 'popsize', 'iterations', 'n_ensembles', 'episode_length']:
		config[key] = int(val)
	elif key in ['load_data', 'hidden_activation', 'data_size', 'save_data', 'script']:
		config[key] = val
	else:
		config[key] = float(val)

'''----------- Environment specific setup --------------'''

test_mismatches = np.array([ [0,0,0] ])
args = ["MinitaurControlledEnv_fastAdapt-v0"]
kwargs = {'control_time_step': config['ctrl_time_step'], 'action_repeat':int(250*config['ctrl_time_step']), 'accurate_motor_model_enabled':0,
		  'pd_control_enabled':1}



if __name__ == "__main__":

	#Define node and rate
	rospy.init_node('fastadapt', anonymous=True)
	rate = rospy.Rate(1000) # 10hz
	robot_utils.init_env_ros()
	time.sleep(1)
	robot_utils.safety_check()
	input("Safety Check OK, Press Enter to Begin ...")
	episode_duration = 2

	main(gym_args=args, gym_kwargs=kwargs, config=config, test_mismatches=test_mismatches,rate =rate,episode_duration =episode_duration)
