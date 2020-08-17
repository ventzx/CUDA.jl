# 多 GPU 协作

多 GPU 协作有不同的方式：可以使用一个或多个线程、进程或系统。虽然所有这些都与 Julia CUDA 工具链兼容，但还有待完善，一些组合的可用性将会得到显著改善。


## 情景 1：每个 GPU 处理一个进程

最简单且是在 Julia 现有的分布式编程设施上最佳的解决方案，即每个进程使用一个GPU。

```julia
# 每个 GPU 产生一个工人
using Distributed, CUDA
addprocs(length(devices()))
@everywhere using CUDA

# 分配设备
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end
```

节点之间的通信应该通过 CPU 进行（CUDA IPC API 以 `CUDA.cuIpcOpenMemHandle` 和 friends 的形式提供，但不能通过高级封装器提供）。

另外，也可以将 [MPI.jl](https://github.com/JuliaParallel/MPI.jl) 与 CUDA-aware MPI 工具一起使用。在这种情况下，`CuArray` 对象可以作为点对点/集体操作的发送/接收缓冲区传递，以避免通过CPU。


## 情景 2：多个 GPU 处理一个进程

与多进程解决方案类似，通过调用 `CUDA.device!` 来切换到特定的 GPU，从而在一个进程中使用多个 GPU。
然而，分配目前并不与 GPU 挂钩，所以应该注意只用分配到当前 GPU 上的数据工作。

!!! warning

   CUDA 内存池还没有设备感知，有效地打破了多 GPU-单进程的并发。
   除非你能支持跨设备的内存操作（例如使用 `cuCtxEnablePeerAccess`），否则不要将这种方法用于严肃的工作。

为了避免这些困难，你可以使用所有设备都能访问的统一内存。
这些 API 可以通过高级包装器获得，但还没有被 `CuArray` 构造函数所公开：

```julia
using CUDA

gpus = Int(length(devices()))

# 生成 CPU 数据
dims = (3,4,gpus)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

# CuArray 还不支持统一内存，所以分配我们自己的缓冲区。
buf_a = Mem.alloc(Mem.Unified, sizeof(a))
d_a = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_a),
                  dims; own=true)
copyto!(d_a, a)
buf_b = Mem.alloc(Mem.Unified, sizeof(b))
d_b = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_b),
                  dims; own=true)
copyto!(d_b, b)
buf_c = Mem.alloc(Mem.Unified, sizeof(a))
d_c = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_c),
                  dims; own=true)
```

这里分配的数据使用 GPU id 作为最外侧的维度，它可以用来提取连续内存的视图，这些视图代表了要被单个 GPU 处理的切片：

```julia
for (gpu, dev) in enumerate(devices())
    device!(dev)
    @views d_c[:, :, gpu] .= d_a[:, :, gpu] .+ d_b[:, :, gpu]
end
```

在下载数据之前，请确保同步设备：

```julia
for dev in devices()
    # 注意：通常情况下，你会使用事件，并等待他们。
    device!(dev)
    synchronize()
end

using Test
c = Array(d_c)
@test a+b ≈ c
```


## 情景 3：每个 GPU 处理一个线程

不建议采用这种方法，因为多线程是最近才被加入 Julia 中的，很多包，包括 Julia GPU 编程的包，都还没有做到线程安全。目前，该工具链模仿了 CUDA 运行库的行为，并在所有设备上使用同一环境。
