# 疑难排解


## CUDA toolkit does not contain `XXX`

这意味着您有一个不完整或丢失的CUDA工具包，或者没有找到工具包的所有必需部分。修复你的 CUDA 工具包安装以确保缺失的二进制文件在你的系统中。 另外，如果在非标准位置安装了 CUDA，请使用 `CUDA_HOME` 环境变量将 Julia 指向该位置。


## UNKNOWN_ERROR(999)

如果你遇到这个错误，目前已知有几种可能导致了它：

-CUDA 驱动程序和驱动程序库之间的不匹配：例如在 Linux上，可以在 `dmesg` 中找到一些端倪。
-CUDA 驱动程序的状态不好：恢复后可能会发生这种情况。试着重新启动。

一般来说，我们不可能说出这个错误的原因，但是这不能怪罪 Julia。 确保你的设置工作正常（例如尝试执行 `nvidia-smi` 等，那是一个CUDA C 二进制文件），如果一切看起来都很好，那么文件就是一个问题。

## NVML library not found (on Windows)

检查并确保 `NVSMI` 文件夹在您的`路径`中。默认情况下可能不是这样。查看 `C:\Program Files\NVIDIA Corporation` 路径下的 `NVSMI` 文件夹，你可以看到里面的 `nvml.dll`。可以将此文件夹添加到`路径`中，并检查 `nvidia-smi` 是否正常运行。

## LLVM error: Cannot cast between two non-generic address spaces

您正在使用 LLVM 的一个未打补丁的副本，这可能是由于使用 Linux 发行版打包的 Julia 造成的。
解决方案通常是使用 LLVM 的全局副本，而不是 Julia 在编译过程中构建并修补的副本。这是不受支持的：LLVM 不能像普通的共享库那样被简单地使用，因为 Julia（和其他 LLVM 用户一样）有一个广泛的补丁列表，可以应用到所支持的特定版本的 LLVM 上。

因此，建议使用官方二进制文件，或者向 Linux 发行版的维护人员建议，使用不设置 `USE_SYSTEM_LLVM=1` 的 Julia 版本。

