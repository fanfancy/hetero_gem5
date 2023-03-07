python3 test_intralayer.py \
	--architecture ours \
	--app_name resnet18 \
	--alg random \
	--encode_type index \
	--dataflow ours \
	--chiplet_num_max 16 \
	--chiplet_num_min 1 \
	--chiplet_parallel  P_stable \
	--PE_parallel All \
	--save_all_records 0 \
	--layer_fuse_tag 0 \
	--optimization_objective edp \
	--temporal_level 3

	