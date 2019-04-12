#!/usr/bin/env python
"""
.. module:: cmd_minitaur_xy
   :synopsis: A rosnode which waits for controller parameters for the minitaur and run episodes by sending commands to /command/xy_cmd

.. moduleauthor:: Eloise Dalin <eloise.dalin@inria.fr>
"""
import rospy
import std_msgs
import sys
import os
import time
import math
from timeit import default_timer as timer
import time
import numpy as np
import tf
from std_srvs.srv import Empty, EmptyResponse
from pyminitaur.srv import Duration, Mode, Ctrl, Distance, GoToNextEpisode
from std_msgs.msg import Float64MultiArray, Float64
from pycontrollers.controller import Controller

def load_centroids(filename):
	""" Load the centroid file from map_elites
	:param filename: name of the centroid file
	:returns: np array containing the centroid points
	"""
	points = np.loadtxt(filename)
	return points

def load_data(filename, dim,dim_ctrl):
	""" Load the archive map from map_elite
	:param filename: name of archive containing the map infromation
	:param dim: dimensionality of the map
	:param dim_ctrl: dimension of the genotype
	:returns: fitness, associated descriptors and genotypes
	"""
	print("Loading ",filename)
	data = np.loadtxt(filename)
	fit = data[:, 0:1]
	desc = data[:,1: dim+1]
	x = data[:,dim+1:dim+1+dim_ctrl]
	return fit, desc, x

def talker():
	global mode, episode_duration, ctrl, controller
	t=0
	dt = 0.001

	#Check robot namespace
	if rospy.has_param('robotname'):
		robotname = rospy.get_param('robotname')
	else:
		robotname = "robot0"
	#Define publisher to send XY commands
	pub = rospy.Publisher(robotname + '/command/xy_cmd', Float64MultiArray, queue_size=50)
	dist_pub = rospy.Publisher(robotname + '/end_episode_travelled_distance', Float64, queue_size=50)

	#Define node and rate
	rospy.init_node('cmd_minintaur_xy', anonymous=True)
	rate = rospy.Rate(1000) # 10hz

	#Define service to start and stop the episode
	#publish service
	#start episode
	def start_episode(req):
		global mode
		mode = 1
		rospy.set_param('mode', 1)
		return EmptyResponse()
	#stop episode
	def stop_episode(req):
		global mode
		mode = 0
		rospy.set_param('mode', 0)
		return EmptyResponse()
	rospy.Service('start_episode', Empty, start_episode)
	rospy.Service('stop_episode', Empty, stop_episode)


	#For ite to be able to set some parameters
	#rosparam are not used because rosparam.get and set is slowing down the ros loop too much, not good for control
	def set_ctrl(req):
		global ctrl, controller
		ctrl = req.ctrl
		controller = Controller(ctrl,1000)
		print(ctrl)
		return True
	#stop episode
	def set_mode(req):
		global mode
		mode = req.mode
		print(mode)
		return True
	#stop episode
	def set_episode_duration(req):
		global episode_duration
		episode_duration = req.duration
		return True
	rospy.Service('set_ctrl', Ctrl, set_ctrl)
	rospy.Service('set_mode', Mode, set_mode)
	rospy.Service('set_episode_duration', Duration, set_episode_duration)
	# rospy.Service('get_distance', Duration, set_episode_duration)

	#tell ite if the episode is still running PUT THAT IN ITE
	# rospy.Service('is_running', Empty, stop_episode)

	#Look at the data published by vrpn
	restart_odom = rospy.ServiceProxy('odom_transform_restart', Empty)
	listener = tf.TransformListener()

	#Load data from the map with input args if some are there
	if len(sys.argv) == 3:
		dim_ctrl=24
		centroids = load_centroids(sys.argv[1])
		fits, descs, ctrls = load_data(sys.argv[2], centroids.shape[1],dim_ctrl)
		print("LOADING CTRL FROM THE MAP")
		print("Max in the map : ", max(fits))
		print("Associated ctrl index : ", np.argmax(fits))
		print("Associated ctrl : ", ctrls[np.argmax(fits)])
		default_ctrl = ctrls[np.argmax(fits)]
	else:
		aX = 0.02
		aY = 0.03
		d = 0.5
		default_ctrl = [aX, 0.1, d, aY, 0.0, d,aX, 0.6, d,aY, 0.5, d,aX, 0.6, d,aY, 0.5, d,aX,0.1,d,aY, 0.0, d]

	ctrl = default_ctrl
	episode_duration = 5
	mode = 0

	curr_distance = 0
	end_episode_travelled_distance = -1000
	controller = Controller(ctrl,1000)

	while not rospy.is_shutdown():
		# print("mode, ", mode)
		# print("ctrl, ", ctrl)
		# print("episode_duration ", episode_duration)

		#Load data from parameters if defined
		# ctrl = rospy.get_param('ctrl', default_ctrl)
		# episode_duration = rospy.get_param('duration', 5)
		# mode = rospy.get_param('mode', 0)
		# episode_duration = 5
		# mode = 1
		#Create cmd  message
		cmd_msg = Float64MultiArray()
		dist_msg = Float64()

		#If the mode is one, the minitaur executes an episode for a duration episode_duration
		if(mode == 1):
			# rospy.set_param('isRunning', True)#set isRunning to truer when an episode is running
			#At the beginning of the episode a odom frame will be created, it is the starting point from which the relative position of the robot will be computed
			if(t==0):
				restart_odom()
			#Compute end legs trajectories from the controller
			traj = controller.step(t)
			#cmd will contain the 8 motor angles to send to the minitaur
			#the order of the angles has to correspond to the motors order, see the official minitaur documentation for this
			cmd = [traj[0]+0.14,traj[2]+0.14,traj[4]+0.14,traj[6]+0.14,traj[1]+0.025,traj[3]+0.025,traj[5]+0.025,traj[7]+0.025]
			# cmd = [traj[0]+0.175,traj[2]+0.175,traj[4]+0.175,traj[6]+0.175,traj[1],traj[3],traj[5],traj[7]]
			t = t + dt
			#Check the relative position of the minitaur according to odom
			try:
				(trans,rot) = listener.lookupTransform("/minitaur", "/odom",rospy.Time(0))
				curr_distance = math.sqrt(trans[0]*trans[0]+trans[1]*trans[1])
				# end_episode_travelled_distance = curr_distance
			except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				# print("CANNOT READ TF PLEASE CHECK VRPN")
				continue
			#Stop the episode after a time span of episode_duration
			if(t>episode_duration):
				mode = 0
				t = 0
				# rospy.set_param('isRunning', False)
				# rospy.set_param('mode', 0)
				end_episode_travelled_distance = curr_distance
				setdist = rospy.ServiceProxy('set_end_episode_distance', Distance)
				try:
					resp1 = setdist(end_episode_travelled_distance)
				except rospy.ServiceException as exc:
					print("Service did not process request: " + str(exc))
				nextep = rospy.ServiceProxy('go_to_next_episode', GoToNextEpisode)
				try:
					resp1 = nextep()
				except rospy.ServiceException as exc:
					print("Service did not process request: " + str(exc))

		else:
			cmd = [0.14,0.14,0.14,0.14,0.025,0.025,0.025,0.025]
		#Fill in and send messages
		# print(cmd)
		cmd_msg.data = cmd
		pub.publish(cmd_msg)
		dist_msg.data = end_episode_travelled_distance
		dist_pub.publish(dist_msg)
		# print("Travelled distance : ", curr_distance)
		rate.sleep()

if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
