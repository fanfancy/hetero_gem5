# TODO List
- [x] 修改了garnet的tester file,对多引擎仿真要做对应的改动 
  wgt_tag=1001; act_tag=1002; out_tag=1003  
- [x] 单引擎：read output partial sum
- [x] code refactory: 统一文件读入hardware参数和conv参数 
- [ ] 增加不同数据流配置？
- [ ] 拓扑：hetero_mesh_nopRouter 参数从脚本中传进来，而不要写死
---
# hetero_gem5

This repo contains 1) gem5 with support of heterogeneous topology, 2) neural network parser and mapping system and 3) prediction function of single-engine & multiple-engine DNN accelerator performance.  
This repo is developed based on gem5 v20.1.0.

## The baseline heterogeneous single-engine accelerator
<div align=center>
<img src="https://github.com/fanfancy/hetero_gem5/blob/main/img/baseline_noc_nop.png" width="800" alt="baseline_arch"/><br/>
</div>

## Special Support for noc+nop simulation
- gem5/configs/topologies/hetero_mesh.py 支持仿真link wdith不同的mesh，实际仿真中无效果。(:exclamation:不再使用了)
- gem5/configs/topologies/hetero_mesh_nopRouter.py 支持仿真noc+nop router,具体配置信息需要在.py中手动配置。脚本中写法如下:
```
	--topology=hetero_mesh_nopRouter \
	--num-cpus= noc_core_num * nop_size + nop_size \
	--num-dirs= noc_core_num * nop_size + nop_size \
```

## File structure
nnparser_ME_hetero：用于支持noc+nop的多引擎架构仿真。
- run_configs.py 用于指定workload拆分及mapping策略&拓扑。
- mesh_hetero.py 带有noc+nop router拓扑的实现。
- mesh.py 普通mesh实现。
- GAGA_gennew_waitall_fensan_2gene.py 多引擎workload拆分、mapping、性能预测、task file生成代码。

nnparser_SE_hetero：用于支持noc+nop的单引擎架构仿真。
- mesh_hetero.py 带有noc+nop router拓扑的实现。
- randomTest_noc_nop.py 随机生成mapping方案并测试，可以在这里配置硬件、卷积参数。
- config.py 通用的不常修改的参数。

粒度探索部分
- run_granularity_model.py 多进程run所有模型的所有层
- randomTest_noc_nop_granularity_model.py 包括探索空间定义和单层探索程序
- single_engine_predict_granularity.py 具体的层内能量和延迟计算（专用于粒度探索）


## Run 多引擎
依据需求修改nn_parser_CCASM_hetero/run_configs.py.
```
cd nnparser_ME_hetero
python3 GAGA_gennew_waitall_fensan_2gene.py
```
对应task file会生成在：nnparser_ME_hetero/task/对应文件名/  
拷贝task file到dnn_task/对应文件名/

在gem5/run_dnn_example.sh中指定拓扑&对应task file路径
```
cd gem5
sh run_dnn_example.sh
```

## Run 单引擎
```
cd nnparser_SE_hetero
python3 randomTest_noc_nop.py.py
```
拷贝对应的task file
```
cd gem5
sh run_singleEngine_baseline_noc.sh
```
可以得到运行一层网络的延迟信息。

## Useful information for experiment
NNbaton buffer配置：4 chiplets, 8 cores, 8 lanes of 8-size vector MAC, 1.5KB O-L1, 800B A-L1, 18KB W-L1 and 64KB A-L2.

NNbaton中不同网络层:  
activation-intensive layer： VGG-16 conv1 (224,224,3)*(3,3,3,64) =(224, 224, 3)  
weight-intensive layer： VGG-16 conv12 (14,14,512) * (3,3,3,512) = (14,14,512)   
large kernel-size layer： ResNet-50 conv1 (224,224,3)*(7,7,3,64) = (112,112, 64) (stride=2)  
