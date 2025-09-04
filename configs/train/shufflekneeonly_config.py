import os

# relative file path to trained model
task_name = "10Action_shufflekneeonly_thigh_p1p2"  #sensor_trainData_valData
model_path = os.path.join("models", "p1+p2_finalmodel.tar")

# relative path to data
data_dirs = [
    "/home/num2/datasets/EXO/Phase1And2_Parsed/Parsed",
    # "/home/num2/datasets/EXO/Phase3_Parsed/Parsed"
]

# corresponding leg (model is not dependent on side)
side = ["r","l"]


# corresponding model input names in dataset (* is substituted with side)
input_names = ["foot_imu_*_gyro_x", "foot_imu_*_gyro_y", "foot_imu_*_gyro_z",  # 0- 2
				"foot_imu_*_accel_x", "foot_imu_*_accel_y", "foot_imu_*_accel_z",#3-5
				"shank_imu_*_gyro_x", "shank_imu_*_gyro_y", "shank_imu_*_gyro_z",# 6
				"shank_imu_*_accel_x", "shank_imu_*_accel_y", "shank_imu_*_accel_z",#9
				"thigh_imu_*_gyro_x", "thigh_imu_*_gyro_y", "thigh_imu_*_gyro_z",#12
				"thigh_imu_*_accel_x", "thigh_imu_*_accel_y", "thigh_imu_*_accel_z",#15
				"insole_*_cop_x", "insole_*_cop_z", "insole_*_force_y",#18
				"hip_angle_*", "hip_angle_*_velocity_filt",#21
				"knee_angle_*", "knee_angle_*_velocity_filt"]#23

sensor_pick = [12,13,14,15,16,17,23,24,]

action_patterns = [
	# === 按论文中重要性排序的动作筛选 ===
	# r"^normal_walk_.*",  # 1. Level ground walk - 最重要（_shuffle、_0-6，两个慢速效果较差）
	r"^normal_walk_.*_(shuffle|0-6|1-2|1-8|2-0|2-5|skip).*",  # 1. Level ground walk - 最重要（_shuffle、_0-6，两个慢速效果较差,排除）
	r"^poses_.*",  # 2. Standing poses
	# r"^dynamic_walk_.*(high-knees|butt-kicks).*",  # 3. Calisthenics (high-knees, butt-kicks)
	# r"^push_.*",  # 4. Push and pull recovery
	r"^jump_.*_(hop|vertical).*",  # 5. Jump in place
	r"^turn_and_step_.*",  # 6. Turn
	r"^cutting_.*",  # 7. Cut
	r"^sit_to_stand_.*",  # 8. Sit and stand
	r"^walk_backward_.*",  # 9. Backwards walk
	r"^weighted_walk_.*",  # 10. Weighted walk
	# r"^lift_weight_.*",  # 11. Lift and place weight
	# r"^tug_of_war_.*",  # 12. Tug of war
	r"^jump_.*_(fb|lateral).*",  # 13. Jump across
	# r"^normal_walk_.*_(2-0|2-5).*",  # 14. Run (2.0和2.5 m/s)
	# r"^dynamic_walk_.*(toe-walk|heel-walk).*",  # 15. Toe and heel walk
	# r"^twister_.*",  # 16. Twister
	# r"^meander_.*",  # 17. Meander
	r"^incline_walk_.*up.*",  # 18. Inclined walk (上坡)
	r"^stairs_.*down.*",  # 19. Stair descent
	# r"^lunges_.*",  # 20. Lunge
	r"^stairs_.*up.*",  # 21. Stair ascent
	r"^incline_walk_.*down.*",  # 22. Declined walk (下坡)
	r"^start_stop_.*",  # 23. Start and stop
	# r"^ball_toss_.*",  # 24. Medicine ball toss
	# r"^obstacle_walk_.*",  # 25. Step over
	r"^squats_.*",  # 26. Squat
	# r"^curb_.*",  # 27. Curb
	# r"^step_ups_.*",  # 28. Step up
]



# corresponding model label names in dataset
label_names = [ "knee_angle_*_moment"]

# intentional model delay (in data points)
model_delays = [10] # hip moment estimates are delayed by 50 ms

# participant masses for normalizing insole forces.
# - NOTE: This is a simplification. Detailed participant masses are provided in the readme of the corresponding dataset.
participant_masses = {
	"BT01": 80.59,
	"BT02": 72.24,
	"BT03": 95.29,
	"BT04": 98.23,
	"BT06": 79.33,
	"BT07": 64.49,
	"BT08": 69.13,
	"BT09": 82.31,
	"BT10": 93.45,
	"BT11": 50.39,
	"BT12": 78.15,
	"BT13": 89.85,
	"BT14": 67.30,
	"BT15": 58.40,
	"BT16": 64.33,
	"BT17": 60.03,
	"BT18": 67.96,
	"BT19": 69.95,
	"BT20": 55.44,
	"BT21": 58.85,
	"BT22": 76.79,
	"BT23": 67.23,
	"BT24": 77.79
}

