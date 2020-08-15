# # 介绍
#
# *这是一篇平和的关于并行化和 Julia GPU 编程的介绍*
#
# [Julia](https://julialang.org/)拥有对 GPU 编程最高等的支持：
# 你可以用高度抽象或得到细粒度的控制，而不需要离开你最爱的编程语言。 
# 这篇指南的目的是帮助 Julia 用户们迈出他们 GPU 计算的第一步。
# 在这片指南中，你将会比较 CPU 和 GPU 实行同一个简单的计算，了解到一些会影响到你得到的性能的因素。
#
# 这篇指南部分地受到了一篇 Mark Harris 的博客的启发 [An Even Easier
# Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/)。 
# 那是一篇使用编程语言 C++ 的更加容易的关于 CUDA 的介绍。
# 你不需要阅读那篇指南，因为这篇会从零开始讲起。



# ## CPU 上的一个简单例子

# 我们来研究一下这个例子，一个 CPU 上的简单计算。

N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             #  # 递增y的每个元素和其相应的元素x。

# 检查后发现我们得到了正确的答案
using Test
@test all(y .== 3.0f0)

# 从“Test Passed”显示后我们知道一切已经就绪。
# 我们使用了 Float32 数据类型，为切换到 GPU 计算做准备：
# GPU 在使用 Float32 类型时比用 Float64 更快（有些时候快得非常多）。

# 这个计算的一个显著特征是，y 的每个元素都使用相同的操作进行更新。
# 这意味着我们可以把它并行化。


# ### CPU 上的并行化

# 首先让我们在 CPU 上进行并行化。
# 我们将创建一个“内核函数”(算法的计算核心)，实现两个版本，第一个是顺序版本：

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)

# 现在是并行的实现版本：

# parallel implementation
function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

# 现在，如果我在一台 JULIA_NUM_THREADS=4，意味着至少有4个核的机器上启动 Julia，我将得到以下结果：

@assert Threads.nthreads() == 4     #src

# ```julia
# using BenchmarkTools
# @btime sequential_add!($y, $x)
# ```

# ```
#   487.303 μs (0 allocations: 0 bytes)
# ```

# versus

# ```julia
# @btime parallel_add!($y, $x)
# ```

# ```
#   259.587 μs (13 allocations: 1.48 KiB)
# ```

#可以看到并行化在性能上有提升，但由于启动线程的开销，这一提升并没有提高到原来的4倍。
# 当使用较大的数组时，这种开销会被大量的“实际工作”“稀释”；随着内核数增加，将表现出接近于线性的提升。
# 相反，对于小数组，并行版本可能比串行版本慢。



# ## 你的第一次 GPU 计算

# ### 安装

# 对于本教程的大部分内容，你需要有一台具有兼容 GPU 且安装 [CUDA](https://developer.nvidia.com/cuda-downloads) 的计算机。
# 您还应该使用 Julia [包管理器](https://docs.julialang.org/en/latest/stdlib/Pkg/)安装以下包：

# ```julia
# pkg> add CUDA
# ```

# 如果这是你的第一次使用，通过测试 CUDA.jl 包来测试你的 GPU 是否在工作是个不错的主意。

# ```julia
# pkg> add CUDA
# pkg> test CUDA
# ```


# ### GPU 上的并行化

# 我们将首先演示使用 CuArray 类型的高级 GPU 计算，而不显式地编写内核函数：

using CUDA

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

# 这里 d 的意思是“设备”，与“主机”相对。现在我们来产生一些增量：

y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

# 语句数组 (y_d) 将 y_d 中的数据移回主机进行测试。
# 如果我们想对它进行基准测试，我们把它放到一个函数中：

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

# ```julia
# @btime add_broadcast!($y_d, $x_d)
# ```

# ```
#   67.047 μs (84 allocations: 2.66 KiB)
# ```

# 最有趣的部分是对 `CUDA.@sync` 的调用。
# CPU 可以分配任务给 GPU，然后去做其他的事情（比如分配*更多的*任务给 GPU），而 GPU 完成它的任务。
# 在 `CUDA.@sync` 块中封装执行会使CPU阻塞，直到队列中的 GPU 任务完成，类似 `Base.@sync` 等待 CPU 任务的方式。
# 如果没有这样的同步，你需要考虑的是启动计算所花费的时间，而不是执行计算的时间。
# 但大多数时候，你不需要显式同步：有许多执行方法，如从 GPU 复制内存到 CPU，隐式同步执行。

# 从这个特殊的计算机和 GPU 中，你可以看到 GPU 的计算速度明显快于单线程的 CPU 计算，
# 而 CPU 多线程的使用使得用 CPU 实现的版本具有竞争力。
# 根据硬件的不同，可能会得到不同的结果。


# ### 编写你的第一个 GPU 核函数

# 使用高级的 GPU 数组函数使得在 GPU 上执行这一计算变得很容易。
# 然而，我们还并不了解底层的工作原理，而这正是本教程的主要目标。
# 那么让我们用 GPU 核函数实现相同的功能：

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# 除了使用 `CuArray`s `x_d` 和 `y_d` 之外，唯一与 GPU 相关的部分是通过 `@cuda` 进行的*核函数启动*。
# 当您第一次发出这个 `@cuda` 语句时，它将编译核函数 (`gpu_add1!`) 用于在GPU上执行。
# 一旦编译，以后的调用会很快。
# 你可以用 ?@cuda` 语句从 Julia 提示中查看 `@cuda` 的扩展。

# 让我们测试一下这个：

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

# ```julia
# @btime bench_gpu1!($y_d, $x_d)
# ```

# ```
#   119.783 ms (47 allocations: 1.23 KiB)
# ```

# 这比上面基于 broadcasting 的版本要*慢得多*。发生了什么？

# ### 分析

# 当您没有获得预期的性能时，通常您的第一步应该是分析代码并查看它在哪里花费了时间。
# 为此，你需要能够运行 NVIDIA的 [`nvprof`tool](https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/) 工具。
# 在 Unix 系统上，这样启动 Julia：
#
# ```sh
# $ nvprof --profile-from-start off /path/to/julia
# ```
#
# 用你的 Julia 二进制文件的路径替换 `/path/to/julia`。
# 注意，我们不会立即启动剖析器，而是调用 CUDA 接口，用 `CUDq.@profile` 手动启动剖析器。
# 这样就排除了编译内核的时间：

bench_gpu1!(y_d, x_d)  # run it once to force compilation
CUDA.@profile bench_gpu1!(y_d, x_d)

# W当我们退出 Julia REPL 时，profiler进程将打印关于执行核函数和接口调用的信息：

# ```
# ==2574== Profiling result:
#             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
#  GPU activities:  100.00%  247.61ms         1  247.61ms  247.61ms  247.61ms  ptxcall_gpu_add1__1
#       API calls:   99.54%  247.83ms         1  247.83ms  247.83ms  247.83ms  cuEventSynchronize
#                     0.46%  1.1343ms         1  1.1343ms  1.1343ms  1.1343ms  cuLaunchKernel
#                     0.00%  4.9490us         1  4.9490us  4.9490us  4.9490us  cuEventRecord
#                     0.00%  4.4190us         1  4.4190us  4.4190us  4.4190us  cuEventCreate
#                     0.00%     960ns         2     480ns     358ns     602ns  cuCtxGetCurrent
# ```

# 您可以看到100%的时间都花在了 `ptxcall_gpu_add1__1` 上。这是当为了这些输入编译 `gpu_add1!` 时，CUDA.jl 分配的那个核函数的名字。
# （你是否创建了多个数据类型的数组，例如 `xu_d = CUDA.fill(0x01, N)`，
# 你也许还见过 `ptxcall_gpu_add1__2` 和其他的。
# 与 Julia 的其他部分一样，您可以定义单个方法，它将在编译时，针对您正在使用的特定数据类型进行专门化。)

# 为了进一步了解情况，使用 `--print-gpu-trace` 运行分析。
# 你也可以用一个包含所有你想要运行的命令的文件路径参数来调用 Julia（包括对 `CUDA.@profile` 的调用）：
#
# ```sh
# $ nvprof --profile-from-start off --print-gpu-trace /path/to/julia /path/to/script.jl
#      Start  Duration   Grid Size   Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream  Name
#   13.3134s  245.04ms     (1 1 1)      (1 1 1)        20        0B        0B  GeForce GTX TIT         1         7  ptxcall_gpu_add1__1 [34]
# ```

# T这里要注意的关键是“Grid Size”和“Block Size”列中的`(1 1 1)`。
# 这些术语稍后会进行解释，但是现在，只要指出这表明这个计算是按顺序运行的就足够了。
# 值得注意的是，用 GPU 进行顺序处理要比用 CPU 慢得多；GPU 的亮点在于大规模并行。


# ### 编写一个平行的 GPU 核函数

# 为了加速内核，我们希望并行化它，这意味着将不同的任务分配给不同的线程。 To facilitate the assignment of work, each CUDA thread gets access
# 为了方便分配工作，每个CUDA线程都可以访问表示自己唯一身份的变量，就像 [`Threads.threadid()`](https://docs.julialang.org/en/latest/manual/parallel-computing/#Multi-Threading-(Experimental)-1) 对CPU线程一样。
# `threadid` 和 `nthreads` 在 CUDA 中的类似物分别被称为 `threadIdx` 和 `blockDim`；
# 其中一个不同点在于，它们返回一个包含字段 x、y 和 z 的三维结构，以简化最多用于三维数组的笛卡尔索引。
# 因此，我们可以通过以下方式分配独特的工作：

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# 注意这里的 `threads=256`，它将工作划分为 256 个用线性模式编号的线程。
# （对于二维数组，我们可以使用 `threads=(16, 16)`，然后`x`和`y`都是相关的。） 

# 现在让我们尝试对它进行测试：

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

# ```julia
# @btime bench_gpu2!($y_d, $x_d)
# ```

# ```
#   1.873 ms (47 allocations: 1.23 KiB)
# ```

# 好多了！

# 但显然，我们还有很长的路要走，以达到最初用 broadcasting 实现的结果。
# 为了做得更好，我们需要更多的并行化。GPU 在*单流多处理器*（SM）上运行的线程数量有限，但与此同时它也有多个 SM。
# 为了充分利用它们，我们需要运行一个具有多个*块*的核函数。
# 我们将这样分配工作:
#
# ![block grid](intro1.png)
#
# 此图[借用了 C/ C++ 库的描述](https://devblogs.nvidia.com/even-easier-introduction-cuda/); 
# 在Julia中，线程和块从1而不是0开始编号。
# 在这个图中，由 256 个线程组成的 4096 块（即 1048576 = 2^20 个线程）确保每个线程只递增一个条目；
# 但是，为了确保可以处理任意大小的数组，我们仍然用了一个循环：

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N/256)

fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# The benchmark:

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
end

# ```julia
# @btime bench_gpu3!($y_d, $x_d)
# ```

# ```
#   67.268 μs (52 allocations: 1.31 KiB)
# ```

# 最后，我们取得了与 broadcast 版本差不多的性能。
# 让我们再次运行 `nvprof` 来确认这个启动配置：
#
# ```
# ==23972== Profiling result:
#    Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream  Name
# 13.3526s  101.22us           (4096 1 1)       (256 1 1)        32        0B        0B  GeForce GTX TIT         1         7  ptxcall_gpu_add3__1 [34]
# ```


# ### 打印

# 在调试时，需要打印一些值是很常见的。 这是通过 `@cuprint` 实现的：

function gpu_add2_print!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@cuda threads=16 gpu_add2_print!(y_d, x_d)
synchronize()

# 注意，打印输出只有在使用 `synchronize()` 同步整个 GPU 时才生成。
# 这和 `CUDA.@sync` 很相似，是 `cudaDeviceSynchronize` 在 CUDA C++中的对应。


# ### 错误处理

# 本介绍的最后一个主题是关于错误的处理。
# 注意，上面的核函数使用了 `@inbounds`，但是没有检查 `y` 和 `x` 是否有相同的长度。
# 如果你的内核不遵守这些范围，你将会陷入严重的错误：

# ```
# ERROR: CUDA error: an illegal memory access was encountered (code #700, ERROR_ILLEGAL_ADDRESS)
# 堆栈跟踪：
#  [1] ...
# ```

# 如果删除 `@inbounds` 注释，则会得到
#
# ```
# ERROR: a exception was thrown during kernel execution.
#        在调试级别 2 上运行 Julia 以跟踪设备堆栈。
# ```

# 正如错误消息所提到的，更高级别的调试信息将产生更详细的报告。
# 让我们运行与 `-g2` 相同的代码：
#
# ```
# ERROR: a exception was thrown during kernel execution.
# 堆栈跟踪：
#  [1] throw_boundserror at abstractarray.jl:484
#  [2] checkbounds at abstractarray.jl:449
#  [3] setindex! at /home/tbesard/Julia/CUDA/src/device/array.jl:79
#  [4] some_kernel at /tmp/tmpIMYANH:6
# ```

# !!! 警告
#
#     在旧的 GPU 上（计算能力低于 `sm_70`），这些错误是致命的，且会有效地杀死 CUDA 环境。
#     在这样的 GPU 上，使用运行在 CPU 上的代码来执行你的“完整性检查”通常是一个好主意，只有当你认为它安全的时候才将计算交给 GPU。



# ## 总结

# 请记住，CUDA 的高级功能通常意味着您不需要担心如何在底层编写核函数。
# 然而，在许多情况下，可以使用聪明的底层操作来优化计算。
# 希望你现在适应了这种大胆的操作。
