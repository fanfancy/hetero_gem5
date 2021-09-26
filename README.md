# TODO List
- [ ] 修改了garnet的tester file,对多引擎仿真要做对应的改动(NOTE:目前的task一定要增加一种data tag才能运行)

wgt_tag =  (int(1001))
act_tag =  (int(1002))
out_tag =  (int(1003))

- [ ] 单引擎：read output partial sum
- [ ] 单引擎：output位于非innerCP点的情况的check
---
# hetero_gem5
## Support on noc+nop simulation
- gem5/configs/topologies/hetero_mesh.py 支持仿真link wdith不同的mesh，实际仿真中无效果。
- gem5/configs/topologies/hetero_mesh_nopRouter.py 支持仿真noc+nop router,具体配置信息需要在.py中手动配置。

## Performance prediction & Genetic algorithm
位于nn_parser_CCASM_hetero文件下：
- nn_parser_CCASM_hetero/run_configs.py 用于指定workload拆分及mapping策略&拓扑。
- nn_parser_CCASM_hetero/mesh_hetero.py 带有noc+nop router拓扑的实现。
- nn_parser_CCASM_hetero/mesh.py 普通mesh实现。

## how to run
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
