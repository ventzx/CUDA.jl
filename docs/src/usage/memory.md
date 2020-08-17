# 内存管理

使用GPU时很重要的一点是管理它的内存。`CuArray` 类型内存管理的主要接口。创建 `CuArray` 时会在 GPU 上分配数据，复制元素到 CuArray 时会上传值到 GPU，将其转换回 `Array` 时会下载值到 CPU：

```julia
# 在 CPU 上生成一些数据
cpu = rand(Float32, 1024)

# 在 GPU 上分配
gpu = CuArray{Float32}(undef, 1024)

# 从 CPU 复制到 GPU
copyto!(gpu, cpu)

# 下载并验证
@test cpu == Array(gpu)
```

完成这些操作的一个较为简单的方法是调用复制构造函数，即 `CuArray(cpu)`。


## 保存类型的上传

在许多情况下，您可能不想将输入转换为密集的 `CuArray` 类型。例如，对于数组包装器，你会希望在 GPU 上保留该包装器类型，只上传其中的数据。[Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) 包正是这样做的，它包含了一个关于如何解包和重构类型（如数组封装）的规则列表，以便我们在诸如将数据上传到 GPU 时可以保留类型：

```julia-repl
julia> cpu = Diagonal([1,2])     # wrapped data on the CPU
2×2 Diagonal{Int64,Array{Int64,1}}:
 1  ⋅
 ⋅  2

julia> using Adapt

julia> gpu = adapt(CuArray, cpu) # upload to the GPU, keeping the wrapper intact
2×2 Diagonal{Int64,CuArray{Int64,1,Nothing}}:
 1  ⋅
 ⋅  2
```

因为这是一个很常见的操作，`cu` 函数可以很方便地帮你完成它：

```julia-repl
julia> cu(cpu)
2×2 Diagonal{Float32,CuArray{Float32,1,Nothing}}:
 1.0   ⋅
  ⋅   2.0
```

!!! 警告

    `cu` 函数是很固执的，坚持将输入标量转换为 `Float32` 类型。
    这通常是一个很好的决定，因为 `Float64` 和许多其他标量类型在 GPU 上的表现很糟糕。
    如果不需要转换，则直接使用 `adapt`。


## 垃圾收集

`CuArray` 类型的实例由 Julia 垃圾收集器管理。这意味着，一旦它们无法被执行，就会被收集起来，释放或重新利用它们占用的内存。不需要手动管理内存，只要确保你的对象是不被执行的（即没有实例或引用）。

### 内存池

在幕后，内存池将保持你的对象并缓存底层内存，以加快今后的分配。因此，你的GPU可能看上去出现了内存耗尽的情况，但实际上并没有。当内存压力较大时，内存池会自动释放缓存对象：

```julia-repl
julia> CUDA.memory_status()             # initial state
Effective GPU memory usage: 10.51% (1.654 GiB/15.744 GiB)
CUDA GPU memory usage: 0 bytes
BinnedPool usage: 0 bytes (0 bytes allocated, 0 bytes cached)

julia> a = CuArray{Int}(undef, 1024);   # allocate 8KB

julia> CUDA.memory_status()
Effective GPU memory usage: 10.52% (1.656 GiB/15.744 GiB)
CUDA GPU memory usage: 8.000 KiB
BinnedPool usage: 8.000 KiB (8.000 KiB allocated, 0 bytes cached)

julia> a = nothing; GC.gc(true)

julia> CUDA.memory_status()             # 8KB is now cached
Effective GPU memory usage: 10.52% (1.656 GiB/15.744 GiB)
CUDA GPU memory usage: 8.000 KiB
BinnedPool usage: 8.000 KiB (0 bytes allocated, 8.000 KiB cached)
```

如果你出于某种原因需要回收所有缓存内存，请调用 `CUDA.reclaim()`：

```julia-repl
julia> CUDA.reclaim()
8192

julia> CUDA.memory_status()
Effective GPU memory usage: 10.52% (1.656 GiB/15.744 GiB)
CUDA GPU memory usage: 0 bytes
BinnedPool usage: 0 bytes (0 bytes allocated, 0 bytes cached)
```

!!! 注

    在执行任何高级 GPU 阵列操作之前，绝不应该请求手动回收内存。分配的功能本身应该被调用到内存池，并在必要时释放任何缓存内存。如果只因没有事先手动回收内存而遇到内存不足的情况，说明出了 bug。

### 避免GC压力

当你的应用程序执行大量的内存操作时，GC 过程中花费的时间可能会显著增加。这种情况比在 CPU 上发生的更多，因为 GPU 的内存往往更小，更经常用完。
当这种情况发生时，CUDA 会调用 Julia 垃圾收集器。垃圾收集器会扫描对象，看看是否可以释放它们以拿回一些 GPU 内存。

为了避免依赖 Julia GC 来释放内存，你可以通过调用 `unsafe_free!` 方法直接通知 CUDA.jl 何时可以释放（或重用）一个配置。
一旦你这样做了，你就不能再使用这个数组了。

```julia-repl
julia> a = CuArray([1])
1-element CuArray{Int64,1,Nothing}:
 1

julia> CUDA.unsafe_free!(a)

julia> a
1-element CuArray{Int64,1,Nothing}:
Error showing value of type CuArray{Int64,1,Nothing}:
ERROR: AssertionError: Use of freed memory
```

### 检测泄漏

如果你认为你有一个内存泄漏，或者你想知道你的 GPU 的 RAM 去了哪里，你可以要求内存池显示所有未分配的内存。 由于跟踪这种情况的成本很高，所以这个功能只有在调试级别 2 或更高的情况下运行 Julia 时才能使用（即使用 `-g2` 参数）当你这样做时，上面的 `memory_status()` 函数将显示额外的信息：

```julia-repl
julia> CuArray([1])
1-element CuArray{Int64,1,Nothing}:
 1

julia> CUDA.memory_status()
Effective GPU memory usage: 8.26% (1.301 GiB/15.744 GiB)
CUDA allocator usage: 8 bytes
BinnedPool usage: 8 bytes (8 bytes allocated, 0 bytes cached)

Outstanding memory allocation of 8 bytes at 0x00007fe104c00000
Stacktrace:
 [1] CuArray{Int64,1,P} where P(::UndefInitializer, ::Tuple{Int64}) at CUDA/src/array.jl:107
 [2] CuArray at CUDA/src/array.jl:191 [inlined]
 [3] CuArray(::Array{Int64,1}) at CUDA/src/array.jl:202
 [4] top-level scope at REPL[2]:1
 [5] eval(::Module, ::Any) at ./boot.jl:331
 [6] eval_user_input(::Any, ::REPL.REPLBackend) at julia/stdlib/v1.4/REPL/src/REPL.jl:86
 [7] macro expansion at julia/stdlib/v1.4/REPL/src/REPL.jl:118 [inlined]
 [8] (::REPL.var"#26#27"{REPL.REPLBackend})() at ./task.jl:358
```

### 环境变量

一些环境变量会影响内存分配器的行为：

- `JULIA_CUDA_MEMORY_POOL`：选择不同的内存池。有几种实现方式可供选择：
  - `binned`（默认）：以 pow2 大小的仓为单位缓存内存。
  - `split`：支持拆分分配的缓存池，旨在减少 Julia 垃圾收集器的压力。
  - `simple`：用于示范的非常简单的缓存层。
  - `none`：完全没有内存池，直接交由 CUDA 分配器处理。
- `JULIA_CUDA_MEMORY_LIMIT`：分配给 GPU 内存的上限，单位为字节。

这些环境变量应该在导入软件包之前设置，在运行时改变它们不会有任何影响。


## 批量迭代器

如果您要处理的数据集太大，无法一次装入 GPU，您可以使用 `CuIterator` 来进行批量操作：

```julia
julia> batches = [([1], [2]), ([3], [4])]

julia> for (batch, (a,b)) in enumerate(CuIterator(batches))
         println("Batch $batch: ", a .+ b)
       end
Batch 1: [3]
Batch 2: [7]
```

对于每一个批次，每一个参数（假如是一个数组形式的参数）都会使用上面的`适配`机制上传到 GPU。之后，通过 `unsafe_free!`，内存被急切地放回 CUDA 内存池中以降低GC压力。
