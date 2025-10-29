import os
import numpy as np
# relative file path to trained model
task_name = "10Action_thighkneeonly_thigh_p1p2"  #sensor_trainData_valData
model_path = "checkpoints/train_baseline_manual_windows_thigh_p1p2_20251017_162149/best_model.tar"

# ========== 滑动窗口配置 ==========
# Sliding Window Configuration for Training
use_sliding_window = True  # Set to True to use sliding window for training, False to use original mode
window_size = 280  # Size of each window (number of time steps)
window_stride = 20  # Stride for sliding window (how many steps to slide)
min_trial_length = 280  # Minimum trial length required (should be >= window_size)

# ========== 模型加载 ==========
# 原始 center 形状: (1, 25, 1)
center = np.array([[[-1.3138794898986816], [1.0175517797470093], [1.0200027227401733], [-3.7354042530059814], [10.356223106384277], [-1.115983247756958], [-2.3052029609680176], [-1.2783820629119873], [4.49326753616333], [-2.2510268688201904], [9.043442726135254], [1.0297681093215942], [0.7113392353057861], [-0.42998769879341125], [0.7253568172454834], [2.6528778076171875], [8.714276313781738], [-0.28184762597084045], [-0.021478787064552307], [0.03748692199587822], [6.203179836273193], [-27.908424377441406], [-0.10620186477899551], [-30.666257858276367], [-0.13483266532421112]]])
# 原始 scale 形状: (1, 25, 1)
scale = np.array([[[64.02909851074219], [71.5345458984375], [141.70074462890625], [8.845449447631836], [6.624897480010986], [4.410506725311279], [38.37006759643555], [68.55548095703125], [122.89159393310547], [5.968291282653809], [4.936514377593994], [2.609605073928833], [21.314773559570312], [45.228126525878906], [81.4139633178711], [3.980790376663208], [4.433416366577148], [1.8335528373718262], [0.19110994040966034], [0.0765497237443924], [5.316965579986572], [27.2789306640625], [60.31145477294922], [27.82839012145996], [107.17118835449219]]])

model_mode = "TCN" #"TCN" or "ConvTimeNet"


# relative path to data
data_dirs = [
    # "/home/num2/datasets/EXO/Phase1And2_Parsed/Parsed",
     "/home/num2/datasets/EXO/Phase3_Parsed/Parsed"
]

# corresponding leg (model is not dependent on side)
side = ["l"]


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
	r"^normal_walk_.*_(1-2|1-8|2-0|2-5|skip).*",  # 1. Level ground walk - 最重要（_shuffle、_0-6，两个慢速效果较差,排除）
	r"^poses_.*",  # 2. Standing poses
	# r"^dynamic_walk_.*(high-knees|butt-kicks).*",  # 3. Calisthenics (high-knees, butt-kicks)
	r"^push_.*",  # 4. Push and pull recovery
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
model_delays = [10,0] # hip moment estimates are delayed by 50 ms


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


vel_filter_cutoff = 5.0
sampling_rate = 200.0
vel_filter_sampling_rate = 200.0
input_rate = 100


# 力矩非线性滤波器配置
enable_torque_nonlinear_filter= True  # 是否启用力矩非线性滤波

# 滤波器参数（可选，不设置则使用默认值）
torque_filter_power_pos= 2.0     # 正向幂次 (默认: 1.5)
torque_filter_power_neg= 2.5      # 负向幂次 (默认: 2.5)
torque_filter_gain_pos= 3       # 正向增益 (默认: 1.8)
torque_filter_gain_neg= 0.8       # 负向增益 (默认: 0.8)
torque_filter_input_limit= 0.4    # 输入限制 (默认: 0.4)
torque_filter_output_limit= 0.6   # 输出限制 (默认: 0.6)