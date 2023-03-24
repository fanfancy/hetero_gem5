# AIMT WORK RECURRENCE  
## ACCELERATOR ARCHITECTURE  
TPU-based scale-up architecture  
each TPU core : 
  * 2 PE arrays
  * each array = 128 * 128 16-bit bfloat MAC units
  * HBM 300GB/s  

scale-up TPU-based core:  
  * 16 PE arrays  
  * 8-bit integar  
  * high-end HBM 450GB/s  
  * 3 decoupled buffers for i, w, o; each buffer has multiple banks
  * double-buffering

## Schedule  
**schedule table**: 
  * layer: 
    * MB(indegree, iters, cycles)
    * CB(indegree, iters, cycles)
    * Post layer (layer id)
  * 通过post layer与indegree来建立神经网络不同层之间的依赖性
  * 通过iters来建立同一层的CB与MB之间的依赖性

**Schedule algorithm**
* RM_C : 片上权重存储空间的剩余量（单位是时间）
* AVL_CB : 