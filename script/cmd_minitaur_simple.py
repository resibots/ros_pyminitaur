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

import rospy
import std_msgs
import sys
import os
import time
import math
from timeit import default_timer as timer
import time
from pycontrollers.controller import Controller
from std_msgs.msg import Float64MultiArray
import numpy as np


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
	if rospy.has_param('robotname'):
		robotname = rospy.get_param('robotname')
	else:
		robotname = "robot0"

	pub = rospy.Publisher(robotname + '/command/xy_cmd', Float64MultiArray, queue_size=50)
	rospy.init_node('minitaur_pub', anonymous=True)
	rate = rospy.Rate(1000)

	dim_ctrl=24
	# if len(sys.argv) < 3:
	# 	sys.exit('Usage: %s centroids_file archive.dat' % sys.argv[0])
	# centroids = load_centroids(sys.argv[1])
	# fits, descs, ctrls = load_data(sys.argv[2], centroids.shape[1],dim_ctrl)
	# print ("Max in the map : ", max(fits))
	# print("Associated ctrl index : ", np.argmax(fits))
	# print("Associated ctrl : ", ctrls[np.argmax(fits)])
	#
	# ctrl = ctrls[np.argmax(fits)]



	aX = 0.02
	aY = 0.03
	d = 0.5
	ctrl_sync_diagonal = [aX, 0.1, d, aY, 0.0, d,aX, 0.6, d,aY, 0.5, d,aX, 0.6, d,aY, 0.5, d,aX,0.1,d,aY, 0.0, d]


	d = 0.2
	aX = 0.02
	aY = 0.05
	#Same phase along Y will lead to a forward sync
	#Same phase along X will lead to a up/down sync
	#ctrl = ["front_left", "back_left", "front_right", "back_right"]
	ctrl_back_front = [aX, 1, d,aY, 0.0, d, aX, 0.1, d, aY, 0.0, d, aX, 1, d,aY, 0.0, d,aX,0.1,d,aY, 0.0, d]
	ctrl = ctrl_back_front

	# aX = 0.02
	# aY = 0.04
	# d = 0.6
	# ctrl_sync_diagonal_jump = [aX, 0.0, d, aY, 1 , d,aX, 0.0, d,aY, 0.1, d,aX, 0.0, d,aY, 0.1, d,aX,0.0,d,aY, 1, d]

	aX = 0.02
	aY = 0.03
	d = 0.5
	ctrl_sync_diagonal_2 = [aX, 0.1, d, aY, 0.0, d,aX, 1, d,aY, 0.5, d,aX, 1, d,aY, 0.5, d,aX,0.1,d,aY, 0.0, d]
	ctrl = ctrl_sync_diagonal_2

	aX = 0.02
	aY = 0.05
	d = 0.2
	b = 0.9
	ctrl_gallop1 = [aX, b+0.01, d, aY, b-0.01, d,aX, b+0.1, d,aY, b, d,aX, b+0.1, d,aY, b, d,aX,0.1,d,aY, 0.0, d]
	ctrl = ctrl_gallop1


	aX = 0.03
	aY = 0.01
	d = 0.1
	ctrl_jump = [aX, 0.1, d, aY,0.0, d,aX, 0.1, d,aY, 0.0, d,aX, 0.1, d,aY, 0.0, d,aX,0.1,d,aY, 0.0, d]


	ctrl = ctrl_sync_diagonal

	controller = Controller(ctrl,1000)
	t= 0
	dt = 0.001
	while not rospy.is_shutdown():
		msg = Float64MultiArray()
		traj = controller.step(t)
		#cmd will contain the 8 motor angles to send to the minitaur
		#the order of the angles has to correspond to the function  ApplyAction of the Minitaur class define in pybullet
		if(t<10):
			cmd = [traj[0]+0.14,traj[2]+0.14,traj[4]+0.14,traj[6]+0.14,traj[1]+0.025,traj[3]+0.025,traj[5]+0.025,traj[7]+0.025]
		else:
			cmd = [0.14,0.14,0.14,0.14,0.025,0.025,0.025,0.025]
			# cmd = [0.175,0.175,0.175,0.175,0.0,0.0,0.0,0.0]
		msg.data = cmd
		t = t +dt
		pub.publish(msg)
		rate.sleep()

if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
