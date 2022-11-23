python multi_network_DSE.py \
	--architecture ours \
	--nn_list resnet18+resnet50+VGG16+vit \
	--chiplet_num 16 \
	--Optimization_Objective edp \
	--BW_Reallocator_tag 1 \
	--layout_mapping_method random \
	--tp_TH 4 \
	--sp_TH 4