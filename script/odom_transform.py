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
import rospy
import tf
from std_srvs.srv import Empty, EmptyResponse

class OdomTransform:
    def __init__(self):
        self.restart = False
        self.playing = False

    def handle_beginning(self, req):
        self.restart = True
        self.playing = False
        return EmptyResponse()

    def loop(self):
        # Get params with defaults
        rate = rospy.get_param('~rate', 100.0) # 100Hz
        world_frame = rospy.get_param('~world_frame', '/world')
        odom_frame = rospy.get_param('~odom_frame', '/odom')
        base_link_frame = rospy.get_param('~base_link_frame', '/Robot_1/base_link')

        # Publish service
        rospy.Service('odom_transform_restart', Empty, self.handle_beginning)

        listener = tf.TransformListener()

        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            while self.restart and not rospy.is_shutdown():
                try:
                    (trans,rot) = listener.lookupTransform(world_frame, base_link_frame, rospy.Time(0))
                    self.restart = False
                    self.playing = True
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            if self.playing:
                br = tf.TransformBroadcaster()
                br.sendTransform(trans, rot,
                                 rospy.Time.now(),
                                 odom_frame,
                                 world_frame)
            rate.sleep()

if __name__ == '__main__':
    # Init ROS node
    rospy.init_node('odom_transform')

    odom_transform_node = OdomTransform()

    odom_transform_node.loop()
