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
from controlled_minitaur_env_meta import *

def talker():
	global mode, episode_duration, ctrl, controller
	t=0
	dt = 0.005

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
	rate = rospy.Rate(200) # 10hz

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

	ctrl = []
	rospy.Service('set_ctrl', Ctrl, set_ctrl)
	rospy.Service('set_mode', Mode, set_mode)
	rospy.Service('set_episode_duration', Duration, set_episode_duration)
	# rospy.Service('get_distance', Duration, set_episode_duration)

	#tell ite if the episode is still running PUT THAT IN ITE
	# rospy.Service('is_running', Empty, stop_episode)

	#Look at the data published by vrpn
	restart_odom = rospy.ServiceProxy('odom_transform_restart', Empty)
	listener = tf.TransformListener()




	episode_duration = 2
	mode = 0

	curr_distance = 0
	end_episode_travelled_distance = -1000
	t = 0
	print("FAST adapt")

	while not rospy.is_shutdown():

		#Create cmd  message
		cmd_msg = Float64MultiArray()
		dist_msg = Float64()

		#If the mode is one, the minitaur executes an episode for a duration episode_duration
		if(mode == 1):
			# rospy.set_param('isRunning', True)#set isRunning to truer when an episode is running
			#At the beginning of the episode a odom frame will be created, it is the starting point from which the relative position of the robot will be computed
			if(t==0):
				i = 0
				restart_odom()
				print("ctrl ", ctrl)
				K = int(episode_duration/dt)
				print("K ", K )
				XY = paramToCartesianTrajectory(ctrl,dt,K)# 200Hz and 400 steps for a total duration of 2 seconds

			x_fl = -XY[0][0]
			y_fl = -XY[1][0]

			x_bl = -XY[0][1]
			y_bl = -XY[1][1]

			x_fr = -XY[0][2]
			y_fr = -XY[1][2]

			x_br = -XY[0][3]
			y_br = -XY[1][3]
			#cmd will contain the 8 motor angles to send to the minitaur
			#the order of the angles has to correspond to the motors order, see the official minitaur documentation for this
			# cmd = [0.14,0.14,0.14,0.14,0.025,0.025,0.025,0.025]
			cmd = [x_fl[i], x_bl[i], x_fr[i], x_br[i], y_fl[i], y_bl[i] , y_fr[i], y_br[i]]
			# cmd = [traj[0]+0.175,traj[2]+0.175,traj[4]+0.175,traj[6]+0.175,traj[1],traj[3],traj[5],traj[7]]
			t = t + dt
			i = i + 1
			print(len(x_fl), " " , i, " : ", cmd)
			#Check the relative position of the minitaur according to odom
			try:
				(trans,rot) = listener.lookupTransform("/minitaur", "/odom",rospy.Time(0))
				euler = tf.transformations.euler_from_quaternion(rot)
				curr_distance = math.sqrt(trans[0]*trans[0]+trans[1]*trans[1])
				state = [trans[0],trans[1],math.sin(-euler[2]),math.cos(-euler[2])]
				# end_episode_travelled_distance = curr_distance
			except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				print("CANNOT READ TF PLEASE CHECK VRPN")
				continue

			#Stop the episode after a time span of episode_duration
			if(i==K):
				mode = 0
				t = 0
				# rospy.set_param('isRunning', False)
				# rospy.set_param('mode', 0)
				setstate = rospy.ServiceProxy('set_state', Ctrl)
				try:
					resp1 = setstate(state)
				except rospy.ServiceException as exc:
					print("Service did not process request: " + str(exc))
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
