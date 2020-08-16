# [概述](@id UsageOverview)

CUDA.jl 包为 CUDA 编程提供了三个相关的不同接口：

- `CuArray` 类型：用于数组编程；
- 本地内核编程能力：在Julia中编写 CUDA 内核：
- CUDA API 包装器：用于与 CUDA 库的低级交互。

大部分 Julia CUDA 编程堆栈都可以只通过 `CuArray` 类型实现，并使用可跨平台的编程模式（如`广播`和其他抽象数组）。
只有当你遇到性能瓶颈或者功能缺失时，你才可能需要编写一个自定义核函数或者使用底层的 CUDA 接口。


## `CuArray` 类型

`CuArray` 类型是工具链的一个重要部分。它主要用于管理 GPU 内存，并将数据从 CPU 中读取和传回 CUP：

```julia
a = CuArray{Int}(undef, 1024)

# essential memory operations, like copying, filling, reshaping, ...
b = copy(a)
fill!(b, 0)
@test b == CUDA.zeros(Int, 1024)

# automatic memory management
a = nothing
```

除了内存管理，还有一系列数组操作可以用来处理数据。
其中包括几个接受其他代码作为参数的高阶操作，如 `map`、`reduce` 或者 `broadcast`。
有了这些，不用实际编写自己的GPU核函数，也能执行类似核函数的操作：

```julia
a = CUDA.zeros(1024)
b = CUDA.ones(1024)
a.^2 .+ sin.(b)
```

如果可能，这些操作将与现有的供应商库（如 CUBLAS 和 CURAND）集成。
例如，如果是支持的类型，矩阵相乘或随机数生成将自动分派到这些高质量的库，否则就返回到通用实现。


## 使用 `@cuda` 进行内核编程

如果一个操作不能用 `CuArray` 现有的功能来表示，或者您需要从 GPU 中榨出最后一滴性能，那么您需要编写自定义核函数。核函数是大量并行执行的函数，通过 `@cuda` 宏启动：

```julia
a = CUDA.zeros(1024)

function kernel(a)
    i = threadIdx().x
    a[i] += 1
    return
end

@cuda threads=length(a) kernel(a)
```

这些核函数用一门熟悉的语言提供了 GPU 必须提供的所有灵活性和性能。
然而，并不是所有的 Julia 语句都被支持：通常情况下，您不能分配内存，封禁 I/O，并且”badly-typed code“将不能编译。
一般的经验法则是，保持核函数简单，逐渐地移植代码，同时不断地验证它仍然按照预期编译和执行。


## CUDA API 包装器

对于 CUDA 的高级使用，可以使用 CUDA.jl 中的驱动程序 API 包装器。常见的操作包括同步 GPU，检查它的属性，启动分析器等。这些操作是低级的，但用了高级构造包装以便使用。例如：

```julia
CUDA.@profile begin
    # code that runs under the profiler
end

# or

for device in CUDA.devices()
    @show capability(device)
end
```

如果没有这样的高级包装器，你可以访问底层的 C API（以 `cu` 为前缀的函数和结构），而不需要退出Julia：

```julia
version = Ref{Cint}()
CUDA.cuDriverGetVersion(version)
@show version[]
```
