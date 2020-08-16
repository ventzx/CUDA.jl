# [概述](@id InstallationOverview)

Julia CUDA 堆栈要求用户有一个功能性的 [NVIDIA 驱动程序](https://www.nvidia.com/Download/index.aspx) 和相应的 [CUDA 工具包](https://developer.nvidia.com/cuda-downloads)。前者应该由您或您的系统管理员安装，而后者可以由Julia使用工件子系统自动下载。



## 平台支持

支持所有三种主要的操作系统：Linux、Windows、macOS。However, that
然而，这种支持取决于 NVIDIA 为您的系统提供的 CUDA 工具包，对 macOS 的后续支持可能很快就会被弃用。

同样，我们支持 x86,，ARM，PPC......只要Julia支持你的平台，且存在 NVIDIA 驱动程序和 CUDA 工具包。然而，主要的开发平台（也是唯一的 CI 系统）是 Linux 上的 x86_64。因此如果您使用的是更奇特的组合，可能会有bug。



## 英伟达驱动

为了使用 Julia GPU 堆栈，你需要为你的系统和 GPU 安装 NVIDIA 驱动程序。
您可以在英伟达的主页上找到[详细指引](https://www.nvidia.com/Download/index.aspx)。

如果您正在使用 Linux，那么应该始终考虑通过发行版的包管理器安装驱动程序。
如果你的驱动程序已经过时或者不支持你的 GPU，你需要从 NVIDIA 的主页上下载一个驱动程序。跟 Linux 一样，你需要一个特定于发行版的包（比如 deb 或 rpm）而不是通用的运行文件选项。

如果您使用的是共享系统，请询问系统管理员如何安装或加载 NVIDIA 驱动程序。一般来说，你应该能够找到并使用 CUDA 驱动程序库，它在 Linux 上叫做  `libcuda.dll` ，在 macOS 上叫做 `libcuda.dylib`，在 Windows 上叫做 `nvcuda64.dll`。您应该还可以执行  `nvidia-smi` 命令，该命令列出了您可以访问的所有可用 GPU。

最后，为了能够使用全部的 Julia GPU 堆栈，你需要有权限来配置 GPU 代码。
在 Linux 上，这意味着加载 `nvidia` 内核模块时配置了 `NVreg_RestrictProfilingToAdminUsers=0` 选项（例如，在`/etc/modprobe.d` 中）。
更多信息请参阅[以下文档](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)。




## CUDA 工具包

提供 CUDA 有两种不同的选择：要么你通过 Julia CUDA 软件包发现的方式，[自己安装](https://developer.nvidia.com/cuda-downloads)，要么你让软件包从工件下载 CUDA。如果您可以使用工件（例如您使用的不是不受支持的平台，或者没有特定的要求），我们建议您这样做：CUDA 工具包与 NVIDIA 驱动程序紧密耦合，在选择使用工件时，会自动考虑兼容性。


### 工件

使用工件是默认选项：调用CUDA.jl，这样当你第一次使用相关接口时，会自动下载 CUDA。您可以通过启用调试日志来检查详细的过程：

```
$ JULIA_DEBUG=CUDA julia

julia> using CUDA

julia> CUDA.version()
┌ Debug: Trying to use artifacts...
└ @ CUDA CUDA/src/bindeps.jl:52
┌ Debug: Using CUDA 10.2.89 from an artifact at /home/tim/Julia/depot/artifacts/93956fcdec9ac5ea76289d25066f02c2f4ebe56e
└ @ CUDA CUDA/src/bindeps.jl:108
v"10.2.89"
```


### 本地安装

如果您的平台无法获得工件，Julia CUDA 包将寻找本地 CUDA 安装。

```
julia> CUDA.version()
┌ Debug: Trying to use artifacts...
└ @ CUDA CUDA/src/bindeps.jl:52
┌ Debug: Could not find a compatible artifact.
└ @ CUDA CUDA/src/bindeps.jl:73

┌ Debug: Trying to use local installation...
└ @ CUDA CUDA/src/bindeps.jl:114
┌ Debug: Found local CUDA 10.0.326 at /usr/local/cuda-10.0/targets/aarch64-linux, /usr/local/cuda-10.0
└ @ CUDA CUDA/src/bindeps.jl:141
v"10.0.326"
```

你可能想要禁止使用工件，例如你的系统有一个优化的 CUDA 安装可供使用。 你可以通过在导入 CUDA.jl 时将环境变量 `JULIA_CUDA_USE_BINARYBUILDER` 设置为 `false` 来实现。

若要排除本地 CUDA 安装发现的故障，可以设置 `JULIA_DEBUG=CUDA`，并查看 CUDA.jl 所在的各种路径的样子。通过设置 `CUDA_ROOT`、`CUDA_ROOT` 或 `CUDA_PATH` 环境变量，可以将包引导到特定的目录。


## 本地编辑

CUDA.jl 对本地编辑很友好：你可以在没有 GPU 的系统上安装、预编译，甚至导入软件包，这在构建映像时是很常见的：

```
$ docker run --rm -it julia

(@v1.5) pkg> add CUDA

(@v1.5) pkg> precompile
Precompiling project...
[ Info: Precompiling CUDA [052768ef-5323-5732-b1bb-66c8b64840ba]

(@v1.5) pkg>
```

在运行时，你显然需要一个兼容 CUDA 的 GPU 以及 CUDA 驱动库来与之对接。通常情况下，该库是从主机系统导入的，例如，使用 `--gpus=all` 标志启动 `docker`。 由于 NVIDIA 本地编辑运行时的工作方式，你还需要定义 `NVIDIA_VISIBLE_DEVICES` 和 `NVIDIA_DRIVER_CAPABILITIES` 环境变量来配置主机驱动程序的哪些部分是可用的。


```
$ docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility julia

julia> using CUDA

julia> CUDA.version()
Downloading artifact: CUDA110
v"11.0.0"
```

请注意，你的图像需要提供 `libgomp`，例如执行 `apt install libgomp1`。

如果你想使用已经提供 CUDA 工具包的镜像，你可以将 `JULIA_CUDA_USE_BINARYBUILDER` 环境变量设置为 `false`，如上文所述。
例如，你可以使用 NVIDIA [NVIDIA 官方 CUDA 图像](https://hub.docker.com/r/nvidia/cuda/)（也不需要你定义 `NVIDIA_VISIBLE_DEVICES` 或 `NVIDIA_DRIVER_CAPABILITIES`）。当然，这些图片并没有预装 Julia。

结合两者，[Julia NGC 图像](https://ngc.nvidia.com/catalog/containers/hpc:julia)同时预装了 Julia 和 CUDA 工具包，以及 CUDA.jl 包，以达到最大的易用性。 
```
$ docker run --rm -it --gpus=all nvcr.io/hpc/julia:v1.2.0

julia> using CuArrays
```

请注意，这个图片的当前版本已经严重过时，但你可以在 [GitHub](https://github.com/maleadt/julia-ngc) 上找到更新的源代码。
