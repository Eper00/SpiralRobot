path_xml_tentacle='//home/tomi/karcsi/assets/simulation/tentacle.xml'

import numpy as np

p1=np.array([0.201,0.225,0.201])
p2=np.array([0.201,0.201,0.225])
p3=np.array([0.225,0.201,0.201])
n_joints = 40
x0 = np.zeros(n_joints*2)  

# az első 24 joint pozíció π/6
x0[:n_joints] = np.pi/6

target_positions = {}

joint_z = np.array([
    0.0143255, 0.034275, 0.052814, 0.0700425, 0.0860535,
    0.1009325, 0.11476, 0.12761, 0.13955, 0.150649,
    0.160962, 0.17055, 0.179453, 0.18773, 0.19542,
    0.202571, 0.2092135, 0.2153875, 0.2211245, 0.2264565
])

r=0.23

