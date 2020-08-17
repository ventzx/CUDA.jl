# 数组编程

```@meta
DocTestSetup = quote
    using CUDA

    import Random
    Random.seed!(0)

    CURAND.seed!(0)
end
```

在 CUDA 数组上进行运算是利用 GPU 强大并行计算能力最简单的方法，它提供了一种数组 `CUArray` 和许多在 GPU 硬件上高效计算的数组。我们将在这一部分简单的示范`CUArray`类型的使用。
由于我们通过在`CuArray`类型上实现现有的 Julia 接口来暴露 CUDA 的功能，所以你应该参考[上游的 Julia 文档](https://docs.julialang.org)来了解这些操作的更多信息。
如果你遇到缺掉的函数或者进行了触发["常量迭代"](@UsageWorkflowScalar)的运算，请看一下[问题追踪](https://github.com/JuliaGPU/CUDA.jl/issues)。如果其中没有相关问题，请创建一个。

记住你总是可以通过调用相关的子模块来调用CUDA的API例如，如果部分随机数接口不能被 CUDA 正常应用，你可以看一下 CUDARAND 文件，有可能可以直接从 `CURAND` 子模块中调用方法。在导入 CUDA 包之后，这些子模块都是可用的


## 建设和初始化

`CuArray` 类型可以提供`抽象数组`接口和通常用于普通数组的方法。这意味着你可以像构建`普通数组`一样构建 `CuArray`。

```julia
julia> CuArray{Int}(undef, 2)
2-element CuArray{Int64,1,Nothing}:
 0
 0

julia> CuArray{Int}(undef, (1,2))
1×2 CuArray{Int64,2,Nothing}:
 0  0

julia> similar(ans)
1×2 CuArray{Int64,2,Nothing}:
 0  0
```

从GPU中提取或写入内存可以构建函数，也可以调用 `copyto!` 函数：

```jldoctest
julia> a = CuArray([1,2])
2-element CuArray{Int64,1,Nothing}:
 1
 2

julia> b = Array(a)
2-element Array{Int64,1}:
 1
 2

julia> copyto!(b, a)
2-element Array{Int64,1}:
 1
 2
```


## 更高层的抽象

GPU 数组编程的真正力量在于 Julia 的高阶数组抽象：一种把用户的代码当作实参（argument），然后进行特殊化处理的操作。有了这些函数，你通常不用自己定制 GPU 核函数。
例如，你可以使用 `map` 或 `broadcast` 函数执行元素积（element-wise）的操作：

```jldoctest
julia> a = CuArray{Float32}(undef, (1,2));

julia> a .= 5
1×2 CuArray{Float32,2,Nothing}:
 5.0  5.0

julia> map(sin, a)
1×2 CuArray{Float32,2,Nothing}:
 -0.958924  -0.958924
```

为了减少数组的维度，CUDA.jl 执行了不同种的 `(map)reduce(dim)`：

```jldoctest
julia> a = CUDA.ones(2,3)
2×3 CuArray{Float32,2,Nothing}:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> reduce(+, a)
6.0f0

julia> mapreduce(sin, *, a; dims=2)
2×1 CuArray{Float32,2,Nothing}:
 0.59582335
 0.59582335

julia> b = CUDA.zeros(1)
1-element CuArray{Float32,1,Nothing}:
 0.0

julia> Base.mapreducedim!(identity, +, b, a)
1×1 CuArray{Float32,2,CuArray{Float32,1,Nothing}}:
 6.0
```

可以使用 `accumulate` 以保留中间值：

```jldoctest
julia> a = CUDA.ones(2,3)
2×3 CuArray{Float32,2,Nothing}:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> accumulate(+, a; dims=2)
2×3 CuArray{Float32,2,Nothing}:
 1.0  2.0  3.0
 1.0  2.0  3.0
```


## 逻辑运算

`CuArray` 也可以用布尔数组作为索引来选择项目：

```jldoctest
julia> a = CuArray([1,2,3])
3-element CuArray{Int64,1,Nothing}:
 1
 2
 3

julia> a[[false,true,false]]
1-element CuArray{Int64,1,Nothing}:
 2
```

建立在这之上的是一些语义学中更高层次的函数：

```jldoctest
julia> a = CuArray([11,12,13])
3-element CuArray{Int64,1,Nothing}:
 11
 12
 13

julia> findall(isodd, a)
2-element CuArray{Int64,1,Nothing}:
 1
 3

julia> findfirst(isodd, a)
1

julia> b = CuArray([11 12 13; 21 22 23])
2×3 CuArray{Int64,2,Nothing}:
 11  12  13
 21  22  23

julia> findmin(b)
(11, CartesianIndex(1, 1))

julia> findmax(b; dims=2)
([13; 23], CartesianIndex{2}[CartesianIndex(1, 3); CartesianIndex(2, 3)])
```


## 数组封装器

在一定程度上，CUDA.jl 也支持标准程序库中广为人知的数组封装器们：

```jldoctest
julia> a = CuArray(collect(1:10))
10-element CuArray{Int64,1,Nothing}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10

julia> a = CuArray(collect(1:6))
6-element CuArray{Int64,1,Nothing}:
 1
 2
 3
 4
 5
 6

julia> b = reshape(a, (2,3))
2×3 CuArray{Int64,2,CuArray{Int64,1,Nothing}}:
 1  3  5
 2  4  6

julia> c = view(a, 2:5)
4-element CuArray{Int64,1,CuArray{Int64,1,Nothing}}:
 2
 3
 4
 5
```

上述相连的 `view` 和 `reshape` 已经被特殊化用以返回 `CuArray` 格式的新对象。其他的封装器，比如下面将要讨论的非相邻视图或 LinearAlgebra 封装器，都是使用他们自己的类型来实现的（比如 `SubArray` 或 `Transpose`）。
这可能会带来问题，因为调用这些带着封装好对象的方法将不会再分派到专门的 `CuArray`。那将导致执行常量循环的应急函数的调用

这些常见的运算，例如广播或矩阵乘法，明白如何解决数组包装器，通过使用 [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) 包。然而还没有完全的解决方案, 存在如有些新数组的包装器还没有覆盖，而且只支持一级包装的问题。
有时，唯一的解决办法是将包装器再次具体化为 `CuArray`。


## 随机数

Base 中用于生成随机数的方便函数在 CUDA 模块中也可以使用。

```jldoctest
julia> CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.74021935
 0.9209938

julia> CUDA.randn(Float64, 2, 1)
2×1 CuArray{Float64,2,Nothing}:
 -0.3893830994647195
  1.618410515635752
```

在幕后，这些随机数来自两个不同的生成器：一个来自 [CURAND](https://docs.nvidia.com/cuda/curand/index.html)，另一个来自 GPUArrays.jl 中定义的内核。对这些生成器的运算是用 Random 标准库的方法实现的。

```jldoctest
julia> using Random

julia> a = Random.rand(CURAND.generator(), Float32, 1)
1-element CuArray{Float32,1,Nothing}:
 0.74021935

julia> using GPUArrays

julia> a = Random.rand!(GPUArrays.global_rng(a), a)
1-element CuArray{Float32,1,Nothing}:
 0.13394515
```

CURAND 还支持生成对数正态和泊松分布的数字。

```jldoctest
julia> CUDA.rand_logn(Float32, 1, 5; mean=2, stddev=20)
1×5 CuArray{Float32,2,CuArray{Float32,1,Nothing}}:
 2567.61  4.256f-6  54.5948  0.00283999  9.81175f22

julia> CUDA.rand_poisson(UInt32, 1, 10; lambda=100)
1×10 CuArray{UInt32,2,Nothing}:
 0x00000058  0x00000066  0x00000061  …  0x0000006b  0x0000005f  0x00000069
```

请注意，只有一部分类型支持这些自定义操作。


## 线性代数

CUDA 来自 [CUBLAS](https://developer.nvidia.com/cublas) 库的线性代数功能是通过实现线性代数标准库中的方法公开的。

```julia
julia> # 启用日志记录以证明使用了 CUBLAS 内核。
       CUBLAS.cublasLoggerConfigure(1, 0, 1, C_NULL)

julia> CUDA.rand(2,2) * CUDA.rand(2,2)
I! cuBLAS (v10.2) function cublasStatus_t cublasSgemm_v2(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called
2×2 CuArray{Float32,2,Nothing}:
 0.295727  0.479395
 0.624576  0.557361
```

Certain operations, like the above matrix-matrix multiplication, also have a native fallback
written in Julia for the purpose of working with types that are not supported by CUBLAS:

```julia
julia> # 启用日志记录以证明没有使用 CUBLAS 内核。
       CUBLAS.cublasLoggerConfigure(1, 0, 1, C_NULL)

julia> CUDA.rand(Int128, 2, 2) * CUDA.rand(Int128, 2, 2)
2×2 CuArray{Int128,2,Nothing}:
 -147256259324085278916026657445395486093  -62954140705285875940311066889684981211
 -154405209690443624360811355271386638733  -77891631198498491666867579047988353207
```

存在于 CUBLAS 中，但（还）未被 LinearAlgebra 标准库中的高级构造所覆盖的操作，可以直接从 CUBLAS 子模块中访问。需要注意的是，你不需要直接调用C语言的封装器（比如`cublasDdot`），因为很多操作也有更高级的封装器可用（比如`dot`）。

```jldoctest
julia> x = CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.74021935
 0.9209938

julia> y = CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.03902049
 0.9689629

julia> CUBLAS.dot(2, x, 0, y, 0)
0.057767443f0

julia> using LinearAlgebra

julia> dot(Array(x), Array(y))
0.92129254f0
```


## 求解器

类似于 [CUSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) 库中的 LAPACK 功能也可以通过 线性代数标准库中的方法访问：

```jldoctest
julia> using LinearAlgebra

julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.740219  0.0390205
 0.920994  0.968963

julia> a = a * a'
2×2 CuArray{Float32,2,Nothing}:
 0.549447  0.719547
 0.719547  1.78712

julia> cholesky(a)
Cholesky{Float32,CuArray{Float32,2,Nothing}}
U factor:
2×2 UpperTriangular{Float32,CuArray{Float32,2,Nothing}}:
 0.741247  0.970725
  ⋅        0.919137
```

其他的操作都是绑定在左分运算符上的：

```jldoctest
julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.740219  0.0390205
 0.920994  0.968963

julia> b = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.925141  0.667319
 0.44635   0.109931

julia> a \ b
2×2 CuArray{Float32,2,Nothing}:
  1.29018    0.942772
 -0.765663  -0.782648

julia> Array(a) \ Array(b)
2×2 Array{Float32,2}:
  1.29018    0.942773
 -0.765663  -0.782648
```



## 稀疏数组

[CUSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) 库的稀疏数组功能主要是通过应用于 `CuSparseArray` 对象的 SparseArrays 包的功能来实现的：

```jldoctest
julia> using SparseArrays

julia> x = sprand(10,0.2)
10-element SparseVector{Float64,Int64} with 4 stored entries:
  [3 ]  =  0.585812
  [4 ]  =  0.539289
  [7 ]  =  0.260036
  [8 ]  =  0.910047

julia> using CUDA.CUSPARSE

julia> d_x = CuSparseVector(x)
10-element CuSparseVector{Float64} with 4 stored entries:
  [3 ]  =  0.585812
  [4 ]  =  0.539289
  [7 ]  =  0.260036
  [8 ]  =  0.910047

julia> nonzeros(d_x)
4-element CuArray{Float64,1,Nothing}:
 0.5858115517433242
 0.5392892841426182
 0.26003585026904785
 0.910046541351011

julia> nnz(d_x)
4
```

对于二维数组，可以使用 `CuSparseMatrixCSC` 和 `CuSparseMatrixCSR`。

非集成的功能可以在 CUSPARSE 子模块中再次直接访问。


## FFTs
[CUFFT](https://docs.nvidia.com/cuda/cufft/index.html) 的功能是与 [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) 包的接口集成在一起的：

```jldoctest
julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.740219  0.0390205
 0.920994  0.968963

julia> using CUDA.CUFFT

julia> fft(a)
2×2 CuArray{Complex{Float32},2,Nothing}:
   2.6692+0.0im   0.65323+0.0im
 -1.11072+0.0im  0.749168+0.0im
```
