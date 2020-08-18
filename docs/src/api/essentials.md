# 必备


## 初始化

```@docs
CUDA.functional(::Bool)
has_cuda
has_cuda_gpu
```


## 全局状态

```@docs
context
context!(::CuContext)
context!(::Function, ::CuContext)
device!(::CuDevice)
device!(::Function, ::CuDevice)
device_reset!
```

如果你有一个维护自己全局状态的库或应用程序，你可能需要对环境或任务切换做出反应：

```@docs
CUDA.attaskswitch
CUDA.atdeviceswitch
CUDA.atdevicereset
```
