using LinearAlgebra
const include_scatter = true
eye(T, m::Int) = Matrix{T}(I,m,m)
trap = eye(Float32, 8) .* 0.033e-3 * include_scatter
println(trap)
println(Void)