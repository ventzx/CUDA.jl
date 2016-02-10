function populate_funmap(funmap, version)
    if version >= 3020
        funmap[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
        funmap[:cuCtxCreate]                = :cuCtxCreate_v2
        funmap[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
        funmap[:cuMemGetInfo]               = :cuMemGetInfo_v2
        funmap[:cuMemAlloc]                 = :cuMemAlloc_v2
        funmap[:cuMemAllocPitch]            = :cuMemAllocPitch_v2
        funmap[:cuMemFree]                  = :cuMemFree_v2
        funmap[:cuMemGetAddressRange]       = :cuMemGetAddressRange_v2
        funmap[:cuMemAllocHost]             = :cuMemAllocHost_v2
        funmap[:cuMemHostGetDevicePointer]  = :cuMemHostGetDevicePointer_v2
        funmap[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
        funmap[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
        funmap[:cuMemcpyDtoD]               = :cuMemcpyDtoD_v2
        funmap[:cuMemcpyDtoA]               = :cuMemcpyDtoA_v2
        funmap[:cuMemcpyAtoD]               = :cuMemcpyAtoD_v2
        funmap[:cuMemcpyHtoA]               = :cuMemcpyHtoA_v2
        funmap[:cuMemcpyAtoH]               = :cuMemcpyAtoH_v2
        funmap[:cuMemcpyAtoA]               = :cuMemcpyAtoA_v2
        funmap[:cuMemcpyHtoAAsync]          = :cuMemcpyHtoAAsync_v2
        funmap[:cuMemcpyAtoHAsync]          = :cuMemcpyAtoHAsync_v2
        funmap[:cuMemcpy2D]                 = :cuMemcpy2D_v2
        funmap[:cuMemcpy2DUnaligned]        = :cuMemcpy2DUnaligned_v2
        funmap[:cuMemcpy3D]                 = :cuMemcpy3D_v2
        funmap[:cuMemcpyHtoDAsync]          = :cuMemcpyHtoDAsync_v2
        funmap[:cuMemcpyDtoHAsync]          = :cuMemcpyDtoHAsync_v2
        funmap[:cuMemcpyDtoDAsync]          = :cuMemcpyDtoDAsync_v2
        funmap[:cuMemcpy2DAsync]            = :cuMemcpy2DAsync_v2
        funmap[:cuMemcpy3DAsync]            = :cuMemcpy3DAsync_v2
        funmap[:cuMemsetD8]                 = :cuMemsetD8_v2
        funmap[:cuMemsetD16]                = :cuMemsetD16_v2
        funmap[:cuMemsetD32]                = :cuMemsetD32_v2
        funmap[:cuMemsetD2D8]               = :cuMemsetD2D8_v2
        funmap[:cuMemsetD2D16]              = :cuMemsetD2D16_v2
        funmap[:cuMemsetD2D32]              = :cuMemsetD2D32_v2
        funmap[:cuArrayCreate]              = :cuArrayCreate_v2
        funmap[:cuArrayGetDescriptor]       = :cuArrayGetDescriptor_v2
        funmap[:cuArray3DCreate]            = :cuArray3DCreate_v2
        funmap[:cuArray3DGetDescriptor]     = :cuArray3DGetDescriptor_v2
        funmap[:cuTexRefSetAddress]         = :cuTexRefSetAddress_v2
        funmap[:cuTexRefGetAddress]         = :cuTexRefGetAddress_v2
        funmap[:cuGraphicsResourceGetMappedPointer] = :cuGraphicsResourceGetMappedPointer_v2
        funmap[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
        funmap[:cuCtxCreate]                = :cuCtxCreate_v2
        funmap[:cuMemAlloc]                 = :cuMemAlloc_v2
        funmap[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
        funmap[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
        funmap[:cuMemFree]                  = :cuMemFree_v2
        funmap[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
        funmap[:cuMemsetD32]                = :cuMemsetD32_v2
    end
    if version >= 4000
        funmap[:cuCtxDestroy]               = :cuCtxDestroy_v2
        funmap[:cuCtxPushCurrent]           = :cuCtxPushCurrent_v2
        funmap[:cuCtxPopCurrent]            = :cuCtxPopCurrent_v2
    end
end
