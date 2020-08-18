# 编译器

## 执行

编译器的主要入口是 `@cuda` 宏：

```@docs
@cuda
```

如果需要，你可以使用一个较低级别的 API，让你监测编译器内核：

```@docs
cudaconvert
cufunction
CUDA.HostKernel
CUDA.version
CUDA.maxthreads
CUDA.registers
CUDA.memory
```


## 反映

如果你想检查生成的代码，你可以使用类似来自 InteractiveUtils 的标准库：

```
@device_code_lowered
@device_code_typed
@device_code_warntype
@device_code_llvm
@device_code_ptx
@device_code_sass
@device_code
```

这些宏作为函数形式也是可用的：

```
CUDA.code_typed
CUDA.code_warntype
CUDA.code_llvm
CUDA.code_ptx
CUDA.code_sass
```

更多信息，请查阅 GPUCompiler.jl 文档。CUDA.jl 中实际上只定义了 `code_sass` 功能。

```@docs
@device_code_sass
CUDA.code_sass
```
