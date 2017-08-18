#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32, Float64, Float32MultiArray, Float64MultiArray
import socket
from Features_realtime import features
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.stats import mode

UDP_IP = "192.168.1.3"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # internet, UDP

sock.bind((UDP_IP, UDP_PORT))

# setup ros publisher to publish joint data acquired from kinect
pub = rospy.Publisher('Activities', Float64, queue_size=20)
rospy.init_node('transmit', anonymous=True)
rate = rospy.Rate(20)

count = 1
rest_time = 60
append_joint_pos = []
classifier = joblib.load('svm_model.pkl')       # load classifier

# loop to accept client connection and receive data
while True:
    data, addr = sock.recvfrom(1024)    # buffer size of 1024 bytes
    # print("%s frame(s) of skeleton joint data received" % count)
    data = (data.decode()).split(',')
    joint_positions = [float(i) for i in data]
    # print("received message: ", joint_positions)

    mytime = rospy.get_time()
    # rospy.loginfo(joint_positions)
    # print(mytime)
    pub.publish(joint_positions)
    rate.sleep()

    # append every odd incoming frame
    if (count % 2) > 0:
        append_joint_pos.append(joint_positions)    # use list to append frames till required amount for classification

    # collect multiple frames and perform feature extraction and real-time classification on the data
    if count == rest_time:
        print("Activity classification started")
        # print(append_joint_pos)
        col_labels = ["headX", "headY", "headZ", "neckX", "neckY", "neckZ", "torsoX", "torsoY", "torsoZ", "lshouldX", "lshouldY", "lshouldZ", "lelbowX", "lelbowY", "lelbowZ",
                      "rshouldX", "rshouldY", "rshouldZ", "relbowX", "relbowY", "relbowZ", "lhipX", "lhipY", "lhipZ", "lkneeX", "lkneeY", "lkneeZ", "rhipX", "rhipY", "rhipZ",
                      "rkneeX", "rkneeY", "rkneeZ", "lhandX", "lhandY", "lhandZ", "rhandX", "rhandY", "rhandZ", "lfootX", "lfootY", "lfootZ", "rfootX", "rfootY", "rfootZ"]
        df = pd.DataFrame(append_joint_pos, columns=col_labels)    # convert joints position list to dataframe

        ext_features = features(df)

        class_result = classifier.predict(ext_features)     # classification result
        pred_prob = classifier.predict_proba(ext_features)  # predicted probability of each class
        # print(pred_prob)
        print(class_result)
        print("Activity recognized = %s" % mode(class_result)[0][0])

        count = 0       # reset count
        del append_joint_pos[:]   # reset list

    count += 1