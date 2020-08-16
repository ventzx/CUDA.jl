# 特许用途

CUDA.jl 的特殊之处在于，开发人员可能希望依靠 GPU 工具链，即使用户可能没有一个 GPU。
在本节中，我们将描述两种不同的使用场景以及如何实现它们。
一个你需要记住的关键是 CUDA.jl **将始终被加载**，这意味着您需要手动检查**包是否正常**。

因为 CUDA.jl 总是加载，即使用户没有一个 GPU 或 CUDA，你也会依靠像其他任何包一样依靠它（不使用，例如 requiress.jl）。
这确保了在安装你的包时，分解器会考虑到对 GPU 堆栈的突发性改变。

如果您无条件使用 CUDA.jl 的功能，在包初始化失败的情况下，您将得到一个运行时错误。例如，在一个没有 CUDA 的系统上:

```julia
julia> using CUDA
julia> CUDA.version()
 ┌ Error: Could not initialize CUDA
│   exception =
│    could not load library "libcuda"
│    libcuda.so: cannot open shared object file: No such file or directory
└ @ CUDA CUDA.jl/src/initialization.jl:99
```

为了避免这种情况，你应该调用 `CUDA.functional()` 检查包是否有效，并在此条件下使用 GPU 功能。让我们用两个场景来说明，一个是需要 GPU 的，另一个是 GPU 可选的。


## 场景 1：需要 GPU

如果你的应用需要一个 GPU，并且它的功能在没有 CUDA 的情况下不能工作，你应该导入必要的包并检查它们是否有效：

```julia
using CUDA
@assert CUDA.functional(true)
```

“`true`”作为一个自变量使 CUDA.jl 显示初始化失败的原因。

如果您正在开发一个包，您唯一要注意的是应该在运行时执行此检查。这确保你的模块总是可以被预编译，即使在一个没有 GPU 的系统上：

```julia
module MyApplication

using CUDA

__init__() = @assert CUDA.functional(true)

end
```

当然，这也意味着你应该避免从全局范围调用 GPU 堆栈，因为这个包可能无效。


## 场景 2：GPU是可选的

如果你的应用程序不需要一个 GPU，并可以在没有 CUDA 包的情况下工作，就需要做出一些权衡。
作为一个例子，让我们定义一个函数，上传一个数组到GPU如果可用:

```julia
module MyApplication

using CUDA

if CUDA.functional()
    to_gpu_or_not_to_gpu(x::AbstractArray) = CuArray(x)
else
    to_gpu_or_not_to_gpu(x::AbstractArray) = x
end

end
```

这是可行的，但不能就这样适应在没有 CUDA 的系统上进行预编译的情况。一种选择是在运行时评估代码：

```julia
function __init__()
    if CUDA.functional()
        @eval to_gpu_or_not_to_gpu(x::AbstractArray) = CuArray(x)
    else
        @eval to_gpu_or_not_to_gpu(x::AbstractArray) = x
    end
end
```

但是，这会导致在运行时进行编译，并且可能会抵消预编译所提供的许多优势。作为替代，你可以使用一个全局标志：

```julia
const use_gpu = Ref(false)
to_gpu_or_not_to_gpu(x::AbstractArray) = use_gpu[] ? CuArray(x) : x

function __init__()
    use_gpu[] = CUDA.functional()
end
```

这种方法的缺点是引入了不确定的数据类型。
