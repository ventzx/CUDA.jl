# 利用 Julia 进行 CUDA 编程

CUDA.jl 是使用 CUDA 的英伟达 GPU 编程的主要进入点。这个包允许使用者进行多个复杂度层次上的使用，从简单易用的数组到手工写的，使用底层的 CUDA 接口的核函数
down to hand-written kernels using low-level CUDA APIs.

如果你有任何问题，请随意使用[Julia slack](https://julialang.slack.com/)，或者 [GPU domain of the Julia Discourse](https://discourse.julialang.org/c/domain/gpu)。


## 快速开始

Julia CUDA 堆栈需要一个基础的 CUDA 设置，它包含了一个驱动程序和一个工具包。
一旦你设置完成，请继续安装 CUDA.jl 包：

```julia
using Pkg
Pkg.add("CUDA")
```

为了确保一切都像预计中那样工作，请尝试加载包裹。如果你有时间，请执行它的测试套件：

```julia
using CUDA

using Pkg
Pkg.test("CUDA")
```

想要更多关于安装过程的细节，请查阅[安装](@refInstallationOverview)章节。为了能更细节地了解工具链，请看一下这本指南里的教程。**我们强烈推荐新用户从[教程简介](@ref)开始**。请阅读[用法](@ref UsageOverview)章节以获得一个对于可用的函数的全面了解。下列的资源你可能也会感兴趣：

- 用 Julia 有效利用 GPU：视频/幻灯片[视频](https://www.youtube.com/watch?v=7Yq1UyncDNc),
  [幻灯片](https://docs.google.com/presentation/d/1l-BuAtyKgoVYakJSijaSqaTL3friESDyTOnU2OLqGoA/)
- Julia 如何在 GPU 上编译：[视频](https://www.youtube.com/watch?v=Fz-ogmASMAE)


## 后记

Julia CUDA 堆栈是许多人们共同努力的结果。以下的用户做出了相当重要的贡献：

- Tim Besard (@maleadt) (主导开发者)
- Valentin Churavy (@vchuravy)
- Mike Innes (@MikeInnes)
- Katharine Hyatt (@kshyatt)
- Simon Danisch (@SimonDanisch)


## 支持和引用

这个生态圈中的许多软件是作为学术研究的一部分而开发的。 如果您愿意支持它，请星标相关的储存库，因为这些参数能帮助我们在未来稳定地集资。如果您在您的研究/教学或其他活动中使用了我们的软件，我们会非常感激如果您引用我们的工作。 
[引用](https://github.com/JuliaGPU/CUDA.jl/blob/master/CITATION.bib) 在这个存储库的顶部的号码布列出了相关的文件。
