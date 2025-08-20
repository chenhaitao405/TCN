action_patterns = [
	# === 按论文中重要性排序的动作筛选 ===
	r"^normal_walk_.*",  # 1. Level ground walk - 最重要
	r"^poses_.*",  # 2. Standing poses
	r"^dynamic_walk_.*(high-knees|butt-kicks).*",  # 3. Calisthenics (high-knees, butt-kicks)
	r"^push_.*",  # 4. Push and pull recovery
	r"^jump_.*_(hop|vertical).*",  # 5. Jump in place
	r"^turn_and_step_.*",  # 6. Turn
	r"^cutting_.*",  # 7. Cut
	r"^sit_to_stand_.*",  # 8. Sit and stand
	r"^walk_backward_.*",  # 9. Backwards walk
	r"^weighted_walk_.*",  # 10. Weighted walk
	r"^lift_weight_.*",  # 11. Lift and place weight
	r"^tug_of_war_.*",  # 12. Tug of war
	r"^jump_.*_(fb|lateral).*",  # 13. Jump across
	r"^normal_walk_.*_(2-0|2-5).*",  # 14. Run (2.0和2.5 m/s)
	r"^dynamic_walk_.*(toe-walk|heel-walk).*",  # 15. Toe and heel walk
	r"^twister_.*",  # 16. Twister
	r"^meander_.*",  # 17. Meander
	r"^incline_walk_.*up.*",  # 18. Inclined walk (上坡)
	r"^stairs_.*down.*",  # 19. Stair descent
	r"^lunges_.*",  # 20. Lunge
	r"^stairs_.*up.*",  # 21. Stair ascent
	r"^incline_walk_.*down.*",  # 22. Declined walk (下坡)
	r"^start_stop_.*",  # 23. Start and stop
	r"^ball_toss_.*",  # 24. Medicine ball toss
	r"^obstacle_walk_.*",  # 25. Step over
	r"^squats_.*",  # 26. Squat
	r"^curb_.*",  # 27. Curb
	r"^step_ups_.*",  # 28. Step up
]