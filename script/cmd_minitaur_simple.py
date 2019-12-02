#! /usr/bin/env python
#| This file is a part of the pyite framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#| Antoine Cully, Jeff Clune, Danesh Tarapore, and Jean-Baptiste Mouret.
#|"Robots that can adapt like animals." Nature 521, no. 7553 (2015): 503-507.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
"""
.. module:: cmd_minitaur_simple
   :synopsis: A rosnode which loads the best minitaur controller parameters found in a map_elites archive  and send the commands to /command/xy_cmd

.. moduleauthor:: Eloise Dalin <eloise.dalin@inria.fr>


"""
from controlled_minitaur_env_meta import *
import rospy
import std_msgs
import sys
import os
import time
import math
from timeit import default_timer as timer
import time
# from pycontrollers.controller import Controller
from std_msgs.msg import Float64MultiArray
import numpy as np
global a


def talker():
	if rospy.has_param('robotname'):
		robotname = rospy.get_param('robotname')
	else:
		robotname = "robot0"

	pub = rospy.Publisher(robotname + '/command/xy_cmd', Float64MultiArray, queue_size=50)
	# dist_pub = rospy.Publisher(robotname + '/end_episode_travelled_distance', Float64, queue_size=50)
	rospy.init_node('minitaur_pub', anonymous=True)

	rate = rospy.Rate(250) #50 Hz


	# SinePolicyAdvanced()


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
	    "data_dir": "data/controlled_minitaur_data",
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


	config["data_dir"] = "/home/minitaur/fast_adapt/minitaur_framework/ros_pyminitaur/script/data/test/"


	'''----------- Environment specific setup --------------'''
	test_mismatches = np.array([ [0,0,0] ])

	gym_args = ["MinitaurControlledEnv_fastAdapt-v0"]
	gym_kwargs = {'control_time_step': config['ctrl_time_step'], 'action_repeat':int(250*config['ctrl_time_step']), 'accurate_motor_model_enabled':0,
			  'pd_control_enabled':1}


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

	'''------------------------Test time------------------------------------'''
	n_test_tasks = len(test_mismatches)
	Data = []
	Likelihoods = []
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


			env=env
			init_state=config["init_state"]
			model=models
			steps= config["episode_length"]
			init_mean=np.zeros(config["sol_dim"])
			init_var=0.01 * np.ones(config["sol_dim"])
			config=config
			last_action_seq = None
			task_likelihoods=task_likelihoods,
			pred_high= high
			pred_low = low
			recorder=recorder
			'''------------------------execute------------------------------------'''
			current_state = copy.copy(env.state) if config['online'] else env.reset()
			print(current_state)
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
				cost_object = Cost(model=model, init_state=current_state[2:], horizon=config["horizon"],
								task_likelihoods=task_likelihoods, action_dim=env.action_space.shape[0], goal=config["goal"],
								pred_high=pred_high, pred_low=pred_low, config=config)
				config["cost_fn"] = cost_object.cost_fn
				optimizer = RS_opt(config)
				# sol = optimizer.obtain_solution(sliding_mean, init_var)
				sol = optimizer.obtain_solution()

				a = sol[0:env.action_space.shape[0]]
				x,y = paramToCartesianTrajectory(a,0.004,125)
				# print("x ", x)
				# print("y ", y)
				next_state, r = 0, 0
				for k in range(config['K']):
					if config["record_video"]:
						recorder.capture_frame()
					next_state, rew, _, _ = env.step(a)
					r += rew

				trajectory.append([current_state[2:].copy(), a.copy(), next_state-current_state, -r])
				current_state = next_state
				traject_cost += -r

			if config["record_video"]:
				recorder.capture_frame()
				recorder.close()
			# return trajectory, traject_cost
			c = traject_cost
			'''------------------------execute------------------------------------'''
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


	# XY= np.load("/home/eloise/ros_ws/src/ros_pyminitaur/script/forward.npy")
	#
	# x_fl = XY[0][0]
	# y_fl = XY[1][0]
	#
	# x_bl = XY[0][1]
	# y_bl = XY[1][1]
	#
	# x_fr = XY[0][2]
	# y_fr = XY[1][2]
	#
	# x_br = XY[0][3]
	# y_br = XY[1][3]
	#### STRAIGHT FORWARD
	y_fl = [-2.28274431e-17,  4.41696016e-03,  8.82821117e-03,  1.32201690e-02, 1.75793830e-02,  2.18926243e-02,  2.61469692e-02,  3.03298771e-02, 3.44292620e-02,  3.84335553e-02,  4.23317602e-02,  4.61134960e-02, 4.97690328e-02,  5.32893154e-02,  5.66659778e-02,  5.98913464e-02, 6.29584352e-02,  6.58609302e-02,  6.85931679e-02,  7.11501048e-02, 7.35272825e-02,  7.57207869e-02,  7.77272049e-02,  7.95435782e-02, 8.11673574e-02,  8.25963554e-02,  8.38287033e-02,  8.48628095e-02, 8.56973224e-02,  8.63310981e-02,  8.67631747e-02,  8.69927522e-02, 8.70191800e-02,  8.68419523e-02,  8.64607102e-02,  8.58752520e-02, 8.50855511e-02,  8.40917805e-02,  8.28943444e-02,  8.14939152e-02, 7.98914761e-02,  7.80883672e-02,  7.60863347e-02,  7.38875822e-02, 7.14948218e-02,  6.89113246e-02,  6.61409690e-02,  6.31882850e-02, 6.00584937e-02,  5.67575402e-02,  5.32921193e-02,  4.96696927e-02, 4.58984961e-02,  4.19875371e-02,  3.79465816e-02,  3.37861297e-02, 2.95173807e-02,  2.51521870e-02,  2.07029981e-02,  1.61827949e-02, 1.16050152e-02,  6.98347183e-03,  2.33226506e-03, -2.33431039e-03,-7.00186080e-03, -1.16559905e-02, -1.62823987e-02, -2.08669738e-02,-2.53958846e-02, -2.98556656e-02, -3.42332964e-02, -3.85162725e-02,-4.26926674e-02, -4.67511847e-02, -5.06811998e-02, -5.44727913e-02,-5.81167604e-02, -6.16046409e-02, -6.49286976e-02, -6.80819159e-02,-7.10579815e-02, -7.38512532e-02, -7.64567277e-02, -7.88699995e-02,-8.10872164e-02, -8.31050320e-02, -8.49205564e-02, -8.65313076e-02,-8.79351632e-02, -8.91303153e-02, -9.01152288e-02, -9.08886050e-02,-9.14493505e-02, -9.17965527e-02, -8.22886337e-02, -8.22186354e-02,-8.19558991e-02, -8.15001491e-02, -8.08512657e-02, -8.00093055e-02,-7.89745275e-02, -7.77474263e-02, -7.63287695e-02, -7.47196395e-02,-7.29214795e-02, -7.09361406e-02, -6.87659304e-02, -6.64136617e-02,-6.38826990e-02, -6.11770026e-02, -5.83011687e-02, -5.52604633e-02,-5.20608509e-02, -4.87090143e-02, -4.52123664e-02, -4.15790531e-02,-3.78179453e-02, -3.39386221e-02, -2.99513421e-02, -2.58670054e-02,-2.16971055e-02, -1.74536714e-02, -1.31492012e-02, -8.79658870e-03,-4.40904232e-03]
	x_fl = [0.18640022, 0.1865152 , 0.18652607, 0.18643421, 0.18624212,0.18595333, 0.18557243, 0.18510496, 0.18455737, 0.18393696,0.18325176, 0.18251045, 0.18172229, 0.18089698, 0.18004458,0.17917539, 0.17829984, 0.17742839, 0.17657141, 0.17573911,0.17494138, 0.17418778, 0.17348736, 0.17284866, 0.17227956,0.17178729, 0.17137828, 0.17105816, 0.17083173, 0.17070285,0.17067448, 0.17074859, 0.17092622, 0.17120737, 0.17159109,0.17207545, 0.17265753, 0.17333347, 0.17409848, 0.17494691,0.17587225, 0.17686721, 0.17792378, 0.17903328, 0.18018646,0.18137356, 0.18258441, 0.18380855, 0.18503529, 0.18625383,0.1874534 , 0.18862331, 0.18975314, 0.1908328 , 0.19185265,0.19280363, 0.19367735, 0.19446618, 0.19516337, 0.19576308,0.19626052, 0.19665194, 0.19693474, 0.19710745, 0.19716977,0.1971226 , 0.19696799, 0.19670913, 0.19635035, 0.19589699,0.19535543, 0.19473293, 0.1940376 , 0.19327832, 0.19246456,0.19160637, 0.19071421, 0.18979884, 0.18887125, 0.18794246,0.18702352, 0.18612531, 0.18525845, 0.18443323, 0.18365949,0.18294654, 0.18230305, 0.181737  , 0.18125561, 0.18086526,0.18057145, 0.18037875, 0.18029077, 0.1803101 , 0.16151539,0.16173478, 0.16205175, 0.16246495, 0.16297209, 0.16356995,0.16425446, 0.16502068, 0.16586286, 0.16677446, 0.16774826,0.16877635, 0.16985025, 0.17096096, 0.17209906, 0.17325478,0.1744181 , 0.17557885, 0.17672681, 0.17785183, 0.17894391,0.17999333, 0.18099075, 0.18192731, 0.18279474, 0.18358545,0.18429262, 0.18491029, 0.18543341, 0.18585794, 0.18618085]
	y_bl = [-2.41298160e-17, -4.66886454e-03, -9.33150041e-03, -1.39735473e-02,-1.85807875e-02, -2.31392384e-02, -2.76352413e-02, -3.20555441e-02,-3.63873759e-02, -4.06185140e-02, -4.47373410e-02, -4.87328919e-02,-5.25948905e-02, -5.63137750e-02, -5.98807122e-02, -6.32876023e-02,-6.65270723e-02, -6.95924608e-02, -7.24777938e-02, -7.51777539e-02,-7.76876421e-02, -8.00033352e-02, -8.21212401e-02, -8.40382447e-02,-8.57516692e-02, -8.72592172e-02, -8.85589291e-02, -8.96491382e-02,-9.05284319e-02, -9.11956176e-02, -9.16496950e-02, -9.18898350e-02,-8.22777039e-02, -8.21113802e-02, -8.17521621e-02, -8.11998500e-02,-8.04544091e-02, -7.95159937e-02, -7.83849762e-02, -7.70619826e-02,-7.55479324e-02, -7.38440827e-02, -7.19520745e-02, -6.98739810e-02,-6.76123566e-02, -6.51702841e-02, -6.25514206e-02, -5.97600398e-02,-5.68010688e-02, -5.36801198e-02, -5.04035145e-02, -4.69782998e-02,-4.34122553e-02, -3.97138911e-02, -3.58924350e-02, -3.19578100e-02,-2.79206008e-02, -2.37920109e-02, -1.95838092e-02, -1.53082683e-02,-1.09780936e-02, -6.60634672e-03, -2.20636174e-03,  2.20834197e-03, 6.62415039e-03,  1.10274466e-02,  1.54047019e-02,  1.97425661e-02, 2.40279524e-02,  2.82481196e-02,  3.23907458e-02,  3.64439964e-02, 4.03965825e-02,  4.42378105e-02,  4.79576212e-02,  5.15466200e-02, 5.49960949e-02,  5.82980264e-02,  6.14450859e-02,  6.44306258e-02, 6.72486606e-02,  6.98938409e-02,  7.23614205e-02,  7.46472184e-02, 7.67475767e-02,  7.86593155e-02,  8.03796868e-02,  8.19063278e-02, 8.32372161e-02,  8.43706263e-02,  8.53050911e-02,  8.60393665e-02, 8.65724025e-02,  8.69033199e-02,  8.70313948e-02,  8.69560491e-02, 8.66768495e-02,  8.61935141e-02,  8.55059261e-02,  8.46141551e-02, 8.35184854e-02,  8.22194503e-02,  8.07178721e-02,  7.90149064e-02, 7.71120900e-02,  7.50113915e-02,  7.27152622e-02,  7.02266877e-02, 6.75492367e-02,  6.46871084e-02,  6.16451735e-02,  5.84290116e-02, 5.50449399e-02,  5.15000351e-02,  4.78021457e-02,  4.39598941e-02, 3.99826695e-02,  3.58806086e-02,  3.16645662e-02,  2.73460746e-02, 2.29372923e-02,  1.84509432e-02,  1.39002466e-02,  9.29883836e-03, 4.66068640e-03]
	x_bl = [0.1970349 , 0.19715238, 0.19715977, 0.19705855, 0.19685135,0.19654192, 0.19613512, 0.1956368 , 0.1950538 , 0.19439383,0.19366538, 0.19287763, 0.19204038, 0.19116387, 0.19025875,0.18933588, 0.18840631, 0.18748108, 0.18657115, 0.18568731,0.18484001, 0.18403933, 0.18329486, 0.18261559, 0.18200987,0.1814853 , 0.18104869, 0.18070598, 0.18046222, 0.18032149,0.1802869 , 0.18036054, 0.16161284, 0.16188113, 0.16224645,0.16270696, 0.16325991, 0.16390166, 0.1646277 , 0.16543267,0.16631042, 0.16725407, 0.16825604, 0.16930813, 0.17040159,0.17152719, 0.17267534, 0.17383612, 0.17499943, 0.17615506,0.17729282, 0.1784026 , 0.17947455, 0.18049911, 0.18146717,0.18237016, 0.18320013, 0.18394987, 0.18461298, 0.18518394,0.18565821, 0.18603224, 0.18630356, 0.18647077, 0.18653359,0.18649285, 0.1863505 , 0.18610955, 0.18577407, 0.18534913,0.18484074, 0.18425579, 0.18360193, 0.18288755, 0.18212163,0.18131366, 0.18047353, 0.17961144, 0.17873776, 0.17786295,0.17699745, 0.17615155, 0.17533532, 0.17455848, 0.17383037,0.17315978, 0.17255494, 0.1720234 , 0.171572  , 0.17120679,0.17093297, 0.17075489, 0.17067596, 0.17069864, 0.17082444,0.17105389, 0.17138651, 0.17182085, 0.1723545 , 0.17298404,0.17370517, 0.17451266, 0.1754004 , 0.17636151, 0.17738832,0.17847248, 0.17960501, 0.18077639, 0.18197666, 0.18319549,0.18442227, 0.18564626, 0.18685666, 0.18804272, 0.18919388,0.19029985, 0.19135078, 0.19233729, 0.19325064, 0.19408282,0.19482662, 0.19547573, 0.19602484, 0.19646967, 0.19680704]
	x_br = x_fl
	y_br = y_fl
	x_fr = x_bl
	y_fr = y_bl

	i = 0
	j = 0
	while not rospy.is_shutdown():
		msg = Float64MultiArray()

		#cmd will contain the 8 motor angles to send to the minitaur
		#the order of the angles has to correspond to the function  ApplyAction of the Minitaur class define in pybullet
		# if(t<10):
		# 	cmd = [traj[0]+0.14,traj[2]+0.14,traj[4]+0.14,traj[6]+0.14,traj[1]+0.025,traj[3]+0.025,traj[5]+0.025,traj[7]+0.025]
		#cmd = [,0.14,0.14,0.14,0.025,0.025,0.025,0.025]
		# if(i < len(x_fr)):
		# cmd = [x_fl[i]+0.28, x_bl[i]+0.28 , x_fr[i]+0.28, x_br[i]+0.28,y_fl[i], y_bl[i] , y_fr[i], y_br[i]]
		# 1 : front left
		# 2 : back left
		# 3 : front right
		# 4 : back left
		# increase theta : cartesian y go toward the back of the robot
		# increase r : the leg deploys farther, the x position is going in the ground direction
		# else:
		#cmd = [0.15,0.14,0.14,0.14,0.050,0.050,0.050,0.025]
		size_cmd =  len(x_fr)
		# cmd = [x_fl[i%size_cmd],0.14,0.14,0.14,y_fl[i%size_cmd],0.050,0.050,0.050]
		# i = 0
		# j = 0
		cmd = [x_fl[j%size_cmd], x_bl[i%size_cmd], x_fr[i%size_cmd], x_br[i%size_cmd],y_fl[j%size_cmd], y_bl[i%size_cmd] , y_fr[i%size_cmd], y_br[i%size_cmd]]
        
		# cmd = [0.175,0.175,0.175,0.175,0.0,0.0,0.0,0.0]
		msg.data = cmd
		pub.publish(msg)
		i = i + 1
		j = j + 1
		rate.sleep()

if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
