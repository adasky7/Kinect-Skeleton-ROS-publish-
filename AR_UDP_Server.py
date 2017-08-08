#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32, Float64, Float32MultiArray, Float64MultiArray
import socket

UDP_IP = "192.168.1.3"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #internet, UDP

sock.bind((UDP_IP, UDP_PORT))

#setup ros publisher to publish joint data acquired from kinect
pub = rospy.Publisher('Activities', Float64, queue_size=20)
rospy.init_node('transmit', anonymous=True)
rate = rospy.Rate(20)


while True:
    data, addr = sock.recvfrom(1024)    #buffer size of 1024 bytes
    #data = struct.unpack('f', data)
    data = (data.decode()).split(',')
    joint_positions = [float(i) for i in data]
    #print("received message: ", joint_positions)

    mytime = rospy.get_time()
    rospy.loginfo(joint_positions)
    print(mytime)
    pub.publish(joint_positions)
    rate.sleep()
    
    
