python multi_network_DSE.py \
	--architecture simba \
	--nn_list resnet18+resnet50+VGG16+vit \
	--chiplet_num 36 \
	--Optimization_Objective edp \
	--BW_Reallocator_tag 1 \
	--tp_TH 8 \
	--sp_TH 10