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
max_alpha=0.9
ds = np.diff(np.concatenate([[0], joint_z]))

kappa_max = max_alpha / ds

print(kappa_max)
# szegmens hossz
z = np.concatenate([[0], joint_z])

# --- görbületmező (EZ ADJA AZ S-ALAKOT) ---
# itt direkt vált előjelet: alsó rész egyik irány, felső másik
#kappa = np.sin(np.linspace(0, 2*np.pi, 21)) * 10
kappa = np.ones(21) * 5  # konstans görbület

x, z_pos = 0.0, 0.0
theta = 0.0

for k in range(21):
    if k > 0:
        ds = joint_z[k-1] - joint_z[k-2] if k > 1 else joint_z[0]

        theta += kappa[k] * ds

        dx = np.sin(theta) * ds
        dz = np.cos(theta) * ds  # <-- EZ A KULCS

        x += dx
        z_pos += dz
    else:
        ds = 0.0
        z_pos = 0.0

    name = f"site_in_{k}_0"
    target_positions[name] = np.array([x, 0.0, z_pos])