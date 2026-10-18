# Sim appearance matched to the real hackathon station (white table, white robot, navy blocks, salmon
# Post-it, front webcam 2026-10-18). Source it BEFORE env.sh / collect.sh / eval.sh:
#     source sim_look_real_station.sh && DATASET_NAME=so101_sim_pickplace_real_look ./collect.sh
# Every value is an environment variable read by so101_pickplace/pickplace_env_cfg.py.
# Rendered colours were calibrated against the real frame (table 0.63, Post-it 0.64/0.30/0.33,
# block 0.07/0.08/0.24) with LIGHT_INTENSITY=150; re-measure if the real lighting or camera changes.

# Front camera (env frame: robot base at the origin, arm faces +X, table top at z=0).
# ~60 cm in front of the base, 30 cm high, tilted ~36 deg down, 1.7 cm left of the robot centre line.
export FRONT_CAM_POS="0.60,-0.017,0.30"
export FRONT_CAM_LOOK_AT="0.185,-0.017,0.0"
export FRONT_CAM_FOCAL_MM=15            # ~70 deg horizontal field of view

# Scene look
export TABLE_RGB="0.85,0.85,0.85"       # plain white table instead of the dark SeattleLab table
export TABLE_X_MIN=-0.02                # table edge right behind the robot base (robot is clamped at the edge)
export FLOOR_RGB="0.08,0.08,0.09"
export ROBOT_RGB="0.85,0.85,0.85"       # printed parts white, servos stay dark
export LIGHT_INTENSITY=150

# Task: stack three blocks (medium 30 mm, then big 40 mm, then small 25 mm) on the pad
export STACK=1

# Block + pad (task text: "stack the medium, then the big, then the small blue block on the pink pad")
export CUBE_COLOR=blue
export CUBE_RGB="0.045,0.045,0.2"
export PAD_COLOR="0.87,0.21,0.25"
export PAD_NAME=pink
export PAD_SIZE_MM=76x127               # 3x5 inch Post-it, long side along Y (sideways in the front image)
