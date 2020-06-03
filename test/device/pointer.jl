capability(device()) >= v"3.2" && @testset "unsafe_cached_load" begin

@testset for T in (Int8, UInt16, Int32, UInt32, Int64, UInt64, Int128, Float32, Float64)
    d_a = CuArray(ones(T))
    d_b = CuArray(zeros(T))
    @test Array(d_a) != Array(d_b)

    ptr_1 = reinterpret(Core.LLVMPtr{T,AS.Global}, pointer(d_1))

    let ptr_a=ptr_a #JuliaLang/julia#15276
        @on_device b[] = unsafe_cached_load(ptr_a)
    end
    @test Array(d_a) == Array(d_b)
end

end
