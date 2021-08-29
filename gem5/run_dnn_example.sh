./build/Garnet_standalone/gem5.opt \
	configs/example/garnet_synth_traffic.py \
	--topology=hetero_mesh_nopRouter \
	--num-cpus=20 \
	--num-dirs=20 \
	--mesh-rows=4 \
	--synthetic=DNN \
	--dnn_task=lenet_rf_ga \
	--link_width_bits=128 \
	--if_debug=0 \
	--network=garnet \
	--inj-vnet=2 \
	--sim-cycles=8000 \
	--injectionrate=0.02


