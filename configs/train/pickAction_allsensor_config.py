import os

task_name = "pickAction_kneeOnly_p1p2"
model_path = os.path.join("models", "p1+p2_finalmodel.tar")

# relative path to data
data_dirs = [
    "/home/num2/datasets/EXO/Phase1And2_Parsed/Parsed",
    # "/home/num2/datasets/EXO/Phase3_Parsed/Parsed"
]

# 使用正则表达式筛选动作
# None 或空列表表示使用所有动作
action_patterns = [
    r"^normal_walk_\d+_.*",      # 所有normal_walk
    r"^incline_walk_.*",          # 所有incline_walk
    r"^stairs_.*",                # 所有stairs
    r"^jump_\d+_.*",              # 所有jump
    # r"^dynamic_walk_.*high-knees.*",  # 只要high-knees的dynamic_walk
    # r".*_on$",                    # 只要所有以_on结尾的（外骨骼开启）
]
# action_patterns = None  # 不筛选，使用所有动作

# corresponding leg (model is not dependent on side)
side = ["r","l"]

# corresponding model input(sensor) names in dataset (* is substituted with side)
input_names = ["foot_imu_*_gyro_x", "foot_imu_*_gyro_y", "foot_imu_*_gyro_z",
				"foot_imu_*_accel_x", "foot_imu_*_accel_y", "foot_imu_*_accel_z",
				"shank_imu_*_gyro_x", "shank_imu_*_gyro_y", "shank_imu_*_gyro_z",
				"shank_imu_*_accel_x", "shank_imu_*_accel_y", "shank_imu_*_accel_z",
				"thigh_imu_*_gyro_x", "thigh_imu_*_gyro_y", "thigh_imu_*_gyro_z",
				"thigh_imu_*_accel_x", "thigh_imu_*_accel_y", "thigh_imu_*_accel_z",
				"insole_*_cop_x", "insole_*_cop_z", "insole_*_force_y",
				"hip_angle_*", "hip_angle_*_velocity_filt",
				"knee_angle_*", "knee_angle_*_velocity_filt"]

sensor_pick = [6,7,8,9,10,11,23,24,]

label_names = ["knee_angle_*_moment"]

# intentional model delay (in data points)
model_delays = [10, 0]

# participant masses for normalizing insole forces
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