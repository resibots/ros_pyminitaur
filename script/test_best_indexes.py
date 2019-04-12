# from copy import *
from math import *
import numpy as np
import sys
from copy import copy
significant_diff = 0.1

#Load the CVT voronoi centroids from input archive
def load_centroids(filename):
	points = np.loadtxt(filename)
	return points

#Load map data from archive
def load_data(filename, dim,dim_ctrl):
	print("Loading ",filename)
	data = np.loadtxt(filename)
	fit = data[:, 0:1]
	desc = data[:,1: dim+1]
	x = data[:,dim+1:dim+1+dim_ctrl]
	return fit, desc, x

def impossible_callback(data):
	global impossible_motion
	impossible_motion = data.data

#print a table with the travelled distances
def print_tab(data,path,ctrls,descs,indexes):
	np.save(path+'distances',np.array(data))
	np.save(path+'ctrls',np.array(ctrls))
	np.save(path+'descs',np.array(descs))
	np.save(path+'indexes',np.array(indexes))
	head = ""
	fmt = ""
	for j in range(0,len(data[0])):
		head = head + "map" + str(j+1) +"\t"
		fmt = fmt + "{:0.2f}\t"
	head = head + "\n"
	fmt = fmt + "\n"
	tab_distances = copy(head)
	for d in range(0,len(data)):
		tmp = copy(data[d])
		if(len(data[d])<len(data[0])):
			for a in range(0,len(data[0])-len(data[d])):
				tmp.append(-1)
		tab_distances = tab_distances + fmt.format(*tmp)
	print(tab_distances)
	filename = 'test_minitaur.dat'
	with open(path+filename, 'w') as f:
		f.write(tab_distances)

if __name__ == "__main__":
	path = sys.argv[1]
	global next_ep, end_episode_distance, impossible_motion
	import rospy
	from std_msgs.msg import UInt32, Float64MultiArray, Bool
	impossible_motion = False
	dim_ctrl = 24
	#Define node and rate
	rospy.init_node('test_maps', anonymous=True)
	rate = rospy.Rate(1000) # 10hz


	#Check robot namespace
	if rospy.has_param('robotname'):
		robotname = rospy.get_param('robotname')
	else:
		robotname = "robot0"
	rospy.Subscriber(robotname + '/state/impossible_motion', Bool, impossible_callback)

	list = range(1,11)
	# list=[1,2,3,5,6,8]
	# list = [1]
	# indexes = [1419,5334,1045,6092,1892,566,3017,2078,5772,5049,6081]
	# list =[1,3]
	episode_duration = 10
	test_max_only = False #Only test the best ctrl in the map
	#Number of ctrls to test in the map
	trials_num = 20
	tested_ctrls = []
	tested_distances = []
	tested_descs = []
	from pyminitaur.srv import Distance, GoToNextEpisode,Duration, Mode, Ctrl
	from std_srvs.srv import Empty, EmptyResponse
	def go_to_next_episode(req):
		global next_ep
		next_ep = True
		# print("Next ep")
		return True
	def set_end_episode_distance(req):
		global end_episode_distance
		end_episode_distance = req.end_episode_distance
		return True
	rospy.Service('go_to_next_episode', GoToNextEpisode, go_to_next_episode)
	rospy.Service('set_end_episode_distance', Distance, set_end_episode_distance)
	for i in list:
		distances = []
		descs_list = []
		ctrl_list = []
		index_list = []
		dist_filter = 1
		torque_filter = 0.7
		dump_period  = 50

		#### LOAD MAP ####################################
		centroids = load_centroids(path+str(i)+"/centroids_40000_16.dat")
		dim_x = 24
		fit, beh, x = load_data(path+str(i)+"/archive_20000.dat", centroids.shape[1],dim_ctrl)
		index = np.argmax(fit)

		# #### RECOVER BEST DIST BEAHVIORS #################

		indexes = np.load(path+str(i)+"/best_index.npy")
		trials_num = 1
		print("Num of indexes : ", len(indexes) )
		te = 0
		centroids = load_centroids(path+str(i)+"/centroids_40000_16.dat")
		if(i in list):
			fits, descs, ctrls = load_data(path+str(i)+"/archive_20000.dat", centroids.shape[1],dim_ctrl)
		for j in range(0,trials_num):
			raw_input("Press enter to begin .....")
			if(impossible_motion == True):
				raw_input("WARNING IMPOSSIBLE MOTION WAS DETECTED, RESTARTING THE ROBOT FOR NEXT TRIAL\n PRESS ENTER TO RESTART")
				restart = rospy.ServiceProxy('restart', Empty)
				try:
					resp = restart()
				except rospy.ServiceException as exc:
					print("Service did not process request: " + str(exc))
				raw_input("Press enter to continue .....")
			if(test_max_only):
				print ("Max in the map in simulation : ", max(fits))
				print("Associated ctrl : ", ctrls[np.argmax(fits)])
				print("Associated ctrl index : ", np.argmax(fits))
				index = np.argmax(fits)
				ctrl_to_test = ctrls[np.argmax(fits)]
			else:
				# te = np.random.randint(0,len(indexes))
				index = indexes[te]
				# index = indexes[te]
				te = te + 1
				ctrl_to_test = ctrls[index]
				ctrl_list.append(ctrl_to_test)
				index_list.append(index)
			descs_list.append(descs[index])
			#set episode duration
			setduration = rospy.ServiceProxy('set_episode_duration', Duration)
			try:
				resp1 = setduration(episode_duration)
			except rospy.ServiceException as exc:
				print("Service did not process request: " + str(exc))
			setctrl = rospy.ServiceProxy('set_ctrl', Ctrl)
			try:
				resp1 = setctrl(ctrl_to_test)
			except rospy.ServiceException as exc:
				print("Service did not process request: " + str(exc))
			setmode = rospy.ServiceProxy('set_mode', Mode)
			try:
				resp1 = setmode(1)
			except rospy.ServiceException as exc:
				print("Service did not process request: " + str(exc))
			next_ep =  False #will be true when an episode running
			while(next_ep == False):
				if(rospy.is_shutdown()):
					try:
						resp1 = setmode(0)
					except rospy.ServiceException as exc:
						print("Service did not process request: " + str(exc))
					sys.exit("ROS IS DOWN")
				else:
					rate.sleep()
			# print("REAL TRAVELLED DISTANCE ", end_episode_distance)
			distances.append(end_episode_distance)
			print("")
			if(impossible_motion == True):
				raw_input("WARNING IMPOSSIBLE MOTION WAS DETECTED, RESTARTING THE ROBOT FOR NEXT TRIAL\n PRESS ENTER TO RESTART")
				restart = rospy.ServiceProxy('restart', Empty)
				try:
					resp = restart()
				except rospy.ServiceException as exc:
					print("Service did not process request: " + str(exc))
			if(test_max_only):
				break
		tested_distances.append(distances)
		tested_ctrls.append(ctrl_to_test)
		tested_descs.append(descs_list)
		print_tab(tested_distances,path,ctrl_list,tested_descs,index_list)
