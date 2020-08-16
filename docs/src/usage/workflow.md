# 工作流

为GPU移植或开发应用程序的典型方法如下:

1. 使用通用数组功能开发一个应用程序，并使用`数组`类型在CPU上测试它。
2. 通过切换到 `CuArray` 类型，将应用程序移植到GPU。
3. 通过切换到 CuArray 类型将你的应用程序移植到 GPU 不允许 CPU 后退（“标量索引”）来查找未实现的或与 GPU 执行不兼容的操作。
4. （可选）使用较低级的、cuda 特有的接口来实现缺少的功能或优化性能。


## [标量索引](@id UsageWorkflowScalar)

为了便于移植代码，`CuArray` 支持执行所谓的“标量代码”，即在 for 循环中一次处理一个元素。考虑到 GPU 的工作方式，这是极其缓慢的，并将抹去使用 GPU 带来的任何性能提升。因此，在执行这种迭代时，您会得到警告：

```julia
julia> a = CuArray([1])
1-element CuArray{Int64,1,Nothing}:
 1

julia> a[1] += 1
┌ Warning: Performing scalar operations on GPU arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`
└ @ GPUArrays GPUArrays/src/indexing.jl:16
2
```

一旦你验证了你的应用程序在GPU上正确执行，你应该不允许使用标量索引，而是使用GPU友好的数组操作：

```julia
julia> CUDA.allowscalar(false)

julia> a[1] .+ 1
ERROR: scalar getindex is disallowed
Stacktrace:
 [1] error(::String) at ./error.jl:33
 [2] assertscalar(::String) at GPUArrays/src/indexing.jl:14
 [3] getindex(::CuArray{Int64,1,Nothing}, ::Int64) at GPUArrays/src/indexing.jl:54
 [4] top-level scope at REPL[5]:1

julia> a .+ 1
1-element CuArray{Int64,1,Nothing}:
 2
```

然而，许多数组操作本身都是使用标量索引实现的。
因此，调用一个看起来对 GPU 友好的数组操作可能会出错：

```julia
julia> a = CuArray([1,2])
2-element CuArray{Int64,1,Nothing}:
 1
 2

julia> var(a)
0.5

julia> var(a,dims=1)
ERROR: scalar getindex is disallowed
```

为了解决这些问题，`CuArray` 的许多数组操作被替换为对 GPU 友好的替代方案。如果你遇到这样的情况，可以看看 CUDA.jl 问题跟踪器，如果还没有的话，可以提交一个 bug 报告。
