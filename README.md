# TODO List
- [ ] 修改了garnet的tester file,对多引擎仿真要做对应的改动 （NOTE：目前的task一定要增加一种data tag才能运行）

  wgt_tag=1001; act_tag=1002; out_tag=1003

- [x] 单引擎：read output partial sum
- [ ] code refactory!
---
# hetero_gem5

This repo contains 1) gem5 with support of heterogeneous topology, 2) neural network parser and mapping system and 3) prediction function of single-engine & multiple-engine DNN accelerator performance.

## Special Support for noc+nop simulation
- gem5/configs/topologies/hetero_mesh.py 支持仿真link wdith不同的mesh，实际仿真中无效果。(不再使用了)
- gem5/configs/topologies/hetero_mesh_nopRouter.py 支持仿真noc+nop router,具体配置信息需要在.py中手动配置。

## File structure
位于nn_parser_CCASM_hetero文件下：
- run_configs.py 用于指定workload拆分及mapping策略&拓扑。
- mesh_hetero.py 带有noc+nop router拓扑的实现。
- mesh.py 普通mesh实现。
- GAGA_gennew_waitall_fensan_2gene.py 多引擎workload拆分、mapping、性能预测、task file生成代码。
- single_engine.py 单引擎性能预测&task file生成（under develop).

## How to run
:bug: 目前没法run，因为garnet源码对单引擎的一些更新

依据需求修改nn_parser_CCASM_hetero/run_configs.py.
```
cd nn_parser_CCASM_hetero
python3 GAGA_gennew_waitall_fensan_2gene.py
```
对应task file会生成在：nn_parser_CCASM_hetero/task/对应文件名/

拷贝task file到dnn_task/对应文件名/

在gem5/run_dnn_example.sh中指定拓扑&对应task file路径
```
cd gem5
sh run_dnn_example.sh
```
