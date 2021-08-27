./build/Garnet_standalone/gem5.opt \
	configs/example/garnet_synth_traffic.py \
	--topology=hetero_mesh \
	--num-cpus=16 \
	--num-dirs=16 \
	--mesh-rows=4 \
	--synthetic=DNN \
	--dnn_task=lenet_rf_ga \
	--link_width_bits=128 \
	--if_debug=0 \
	--network=garnet \
	--inj-vnet=2 \
	--sim-cycles=8000 \
	--injectionrate=0.02


