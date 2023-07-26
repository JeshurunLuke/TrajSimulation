



#Consts




module Consts
export C
const C = Dict("c" => 2.99798458E8, "hbar" => 1.054571E-34, "kb" => 1.3806452E-23, "e0" => 8.85418E-12, "q_c" => 1.60217663e-19, "m_e" => 9.11e-31)
end
module atom_class


export cesiumAtom, Energy, getSaturationIntensity, getTransitionRate, atomInterface, getw0, getLifetime, rubidiumAtom


using PyCall
np = pyimport("numpy")
arc = pyimport("arc")


struct atomInterface
   atom
end
function findStates(Atom::atomInterface, States, F::T, mF::T)::Float64 where {T<:AbstractFloat}
   ind = findall(States[2] .== F .&& States[3] .== mF)
   if length(ind) != 1
       return "Error"
   end
   Energy::Matrix{Float64} = States[1]
   return Energy[ind][1]
end


function Energy(Atom::atomInterface, n::T, l::T, j::G, f::G, mF::G, B::G) where {T<:Integer,G<:AbstractFloat}
   ev2Hz = 2.417989242E14
   States = Atom.atom.breitRabi(n, l, j, np.array([B]))
   return findStates(Atom, States, f, mF) + Atom.atom.getEnergy(n, l, j) * ev2Hz
end
function getw0(Atom::atomInterface, stateI::T, stateF::T, B) where {T}
   return Energy(Atom, stateF..., B) - Energy(Atom, stateI..., B)
end
function getSaturationIntensity(Atom::atomInterface, stateI::T, stateF::T) where {T}
   return Atom.atom.getSaturationIntensity(stateI..., stateF...)
end
function getTransitionRate(Atom::atomInterface, stateI::T, stateF::T) where {T}
   return Atom.atom.getTransitionRate(stateF[1:3]..., stateI[1:3]...)
end


function getLifetime(Atom::atomInterface, stateF)
   Atom.atom.getStateLifetime(stateF[1:3]...)
end




Energy(Atom::atomInterface, n::T, l::T, j, f::T, mF::T, B::T) where {T<:Integer} = Energy(Atom, n, l, float(j), float(f), float(mF), float(B))
Energy(Atom::atomInterface, n::T, l::T, j::G, f::T, mF::T, B::G) where {G<:AbstractFloat,T<:Integer} = Energy(Atom, n, l, float(j), float(f), float(mF), float(B))


cesium = arc.Cesium
rubidium = arc.Rubidium87


const cesium_a = cesium()
const rubidium_a = rubidium()


const cesiumAtom = atomInterface(cesium_a)
const rubidiumAtom = atomInterface(rubidium_a)


end
# Utils
module utils
using LinearAlgebra
using Random
using Statistics
using Distributions


export pol_v, decompose_spherical, project_onto_vector, project_onto_plane, rotation_matrix, get_angle, getSphereVec


const pol_v = Dict("sigma_p" => -1 / sqrt(2) * ComplexF64[1, 1im, 0], "sigma_m" => 1 / sqrt(2) * ComplexF64[1, -1im, 0], "pi" => ComplexF64[0, 0, 1])




function decompose_spherical(pol)
   return ComplexF64[dot(conj(pol_v["pi"]), pol), dot(conj(pol_v["sigma_p"]), pol), dot(conj(pol_v["sigma_m"]), pol)]
end


project_onto_vector(v, n) = dot(v, n / norm(n)) * (n / norm(n))
project_onto_plane(x, n) = x - dot(x, n / norm(n)) * (n / norm(n))




function rotation_matrix(axis, theta)
   axis = axis / norm(axis)
   a = cos(theta / 2)
   b, c, d = -axis * sin(theta / 2)
   return [a*a+b*b-c*c-d*d 2*(b*c-a*d) 2*(b*d+a*c)
       2*(b*c+a*d) a*a+c*c-b*b-d*d 2*(c*d-a*b)
       2*(b*d-a*c) 2*(c*d+a*b) a*a+d*d-b*b-c*c]
end


get_angle(v1, v2) = acos(dot(v1, v2) / (norm(v1) * norm(v2)))


function getSphereVec(n)
   uvec = rand(Normal(), n)
   uvec = uvec / norm(uvec)
   return uvec
end


end
# Beam Class
module BeamClass
using ..utils
using LinearAlgebra
export BeamProperties, GaussianBeam, get_Intensity


struct BeamProperties{T}
   Loc::T
   dir::T
   pol
   waist::Float64
   zr::Float64
end


BeamProperties(Loc::T, dir::T, pol) where {T} = BeamProperties{T}(Loc, dir, pol, 1e10, 1e10)
BeamProperties(Loc::T, dir::T, waist::Float64, wavelength::Float64) where T = BeamProperties{T}(Loc, dir, pol_v["pi"], waist, pi * waist^2 / wavelength)




struct GaussianBeam
   beamStruct::BeamProperties
   I0::Float64
   waist::Float64
end


function get_Intensity(Beam::GaussianBeam, pos)
   BP = Beam.beamStruct
   r = norm(project_onto_plane(pos, BP.dir))
   z = norm(project_onto_vector(pos, BP.dir) - BP.Loc)
   w(z_i) = BP.waist * sqrt(1 + (z_i / BP.zr)^2)
   I = Beam.I0 * (BP.waist / w(z))^2 * exp(-2 * r^2 / (w(r)^2))
   return I
end


end






###
module CoolTrap


const c_s = 2.99798458E8
const ħ = 1.054571E-34
const kb = 1.3806452E-23
const ϵ0 =  8.85418E-12
const q_c = 1.60217663e-19
const m_e = 9.11e-31


using ..BeamClass
using ..utils
using Interpolations


using LinearAlgebra
using Statistics: mean, std
using Distributions


using ..atom_class


export MOT_Beam, get_Fnet, Tweezer, Tweezer_WF, generateAtoms, sampleVelocities_BD, Environment_T, get_Fnet_S, get_scatteringrate, get_scatteringrate_abs, get_momSpon, get_momAbs, AbstractTweezer


struct Environment_T
   Intensity
   B_Field
   grid
end




# MOT FUNCTIONS
struct MOT_Beam{T}
   ω0::T
   detuning::T
   k_vec::Vector{T}
   Γ::T
   Γ_spont #  SE::SpontEmission
   Isat::T
   Beam::GaussianBeam
   s0
   environment
   dir
end


function MOT_Beam(Atom::atomInterface, Beam::BeamProperties, environment, stateI, stateF, detuning, I, updateS0=true)
   B_Field_func = environment.B_Field


   ω0 = 2 * pi * getw0(Atom, stateI, stateF, norm(B_Field_func(0, 0, 0)))
   BeamGaussian = GaussianBeam(Beam, I, ω0 + detuning)


   wavelength = c_s / (ω0 / (2 * pi))
   k_vec = 2 * pi / wavelength * Beam.dir
   Γ = getTransitionRate(Atom, stateI, stateF)
   Γ_spont = 1/getLifetime(Atom, stateF)
   Isat = getSaturationIntensity(Atom, stateI, stateF)
   q = stateF[end] - stateI[end]


   rot_mat(pos) = rotation_matrix(cross(B_Field_func(pos...), Beam.dir), get_angle(B_Field_func(pos...), Beam.dir))
   rotated_pol(pos) = rot_mat(pos) * Beam.pol
   decomposed_pol(pos) = decompose_spherical(rotated_pol(pos))
   s0 = pos -> 0


   if !updateS0
       println("Fixing")
       setpoint = get_Intensity(BeamGaussian, [0.0, 0.0, 0.0]) / Isat * abs(decomposed_pol([0.0, 0.0, 0.0])[q >= 0 ? q + 1 : end + 1 + q])^2 / norm(decomposed_pol([0.0, 0.0, 0.0]))^2
       s0 = pos -> setpoint
   else
       println("Dynamic")
       s0 = pos -> get_Intensity(BeamGaussian, pos) / Isat * abs(decomposed_pol(pos)[q >= 0 ? q + 1 : end + 1 + q])^2 / norm(decomposed_pol(pos))^2
   end


   return MOT_Beam(ω0, detuning, k_vec, Γ, Γ_spont,  Isat, BeamGaussian, s0, environment, Beam.dir)
end






function get_scatteringrate_abs(MBeam::MOT_Beam, pos, vel, detuningOffset)
   return MBeam.s0(pos) * MBeam.Γ / 2 * 1 / (1 + MBeam.s0(pos) + (2 * (MBeam.detuning + detuningOffset + dot(MBeam.k_vec, vel)) / MBeam.Γ)^2)
end
function get_scatteringrate(MBeam::MOT_Beam, pos, detuningOffset)
   return MBeam.s0(pos) * MBeam.Γ / 2 * 1 / (1 + MBeam.s0(pos) + (2 * (MBeam.detuning + detuningOffset) / MBeam.Γ)^2)
end
function get_Fabs(MBeam::MOT_Beam, pos, vel, detuningOffset)
   return ħ * (MBeam.ω0 + MBeam.detuning) / c_s* get_scatteringrate_abs(MBeam, pos, vel, detuningOffset) * MBeam.dir
end
function get_momSpon(MBeam::MOT_Beam)
   return ħ * norm(MBeam.k_vec) * getSphereVec(3)
end
function get_momAbs(MBeam::MOT_Beam)
   #println(MBeam.dir)
   return ħ * norm(MBeam.k_vec) * MBeam.dir
end




function get_Fnet(MBeam::MOT_Beam, pos, vel, detuningFunc::Any)
   detuningOffset = detuningFunc(pos...)
   #Spont = get_Fspon(MBeam, pos, detuningOffset)
   Abs = get_Fabs(MBeam, pos, vel, detuningOffset)
   return Abs #+ Spont
end


get_scatteringrate_abs(MBeam::MOT_Beam, pos, vel) = get_scatteringrate_abs(MBeam::MOT_Beam, pos, vel, 0)
get_scatteringrate(MBeam::MOT_Beam, pos) = get_scatteringrate(MBeam::MOT_Beam, pos, 0)
get_Fabs(MBeam::MOT_Beam, pos, vel) = get_Fabs(MBeam::MOT_Beam, pos, vel, 0)
#get_Fspon(MBeam::MOT_Beam, pos) = get_Fspon(MBeam::MOT_Beam, pos, 0)
get_Fnet(MBeam::MOT_Beam, pos, vel) = get_Fnet(MBeam::MOT_Beam, pos, vel, 0)








function get_Fnet(MBeam::MOT_Beam, pos, vel, detuningOffset::Number)
   #Spont = get_Fspon(MBeam, pos, detuningOffset)
   Abs = get_Fabs(MBeam, pos, vel, detuningOffset)
   return Abs #+ Spont
end




get_Fnet_S(MBeam::MOT_Beam, pos, vel, detuningFunc::Any) = vcat(get_Fnet(MBeam, pos, vel, detuningFunc), [get_scatteringrate(MBeam, pos, detuningFunc(pos...))])
get_Fnet_S(MBeam::MOT_Beam, pos, vel, detuningNum::Number) = vcat(get_Fnet(MBeam, pos, vel, detuningNum), [get_scatteringrate(MBeam, pos,detuningNum)])






abstract type AbstractTweezer end


# Tweezer Function
struct Tweezer{T} <: AbstractTweezer
   trapR::T
   trapZ::T
   Beam::GaussianBeam
   ω0::T
   I0::T
   mass::T
   α
end


struct Tweezer_WF{T} <: AbstractTweezer
   trapR
   trapZ
   Beam::GaussianBeam
   mass
   ForceInterp
   α::T
end


function np_gradient(f::AbstractArray{T,N}; h=1.0) where {T,N}
   g = ntuple(N) do dim
       grad = similar(f)
       ax = axes(f, dim)
       # forward difference on the left boundary
       I1 = first(ax)
       I2 = I1 + 1
       for I in CartesianIndices(axes(f))
           if I[dim] < size(f, dim)
               grad[I] = (f[I + CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))] - f[I]) / h
           end
       end
       # central difference in the interior points
       for i in ax[2:end-1]
           I1 = i - 1
           I2 = i + 1
           for I in CartesianIndices(axes(f))
               if I[dim] > 1 && I[dim] < size(f, dim)
                   grad[I] = (f[I + CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))] - f[I - CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))]) / (2h)
               end
           end
       end
       # backward difference on the right boundary
       I1 = last(ax) - 1
       I2 = last(ax)
       for I in CartesianIndices(axes(f))
           if I[dim] > 1
               grad[I] = (f[I] - f[I - CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))]) / h
           end
       end
       grad
   end
   g
end


function Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ; α::Number)
   mass = Atom.atom.mass

   waist = trapR / trapZ * (1 / (pi * sqrt(2))) * wavelength_tweezer
   zr = pi * waist^2 / wavelength_tweezer
   U0 = 1 / 4 * mass * waist^2 * trapR^2
   wavelength_transition = abs(Atom.atom.getTransitionWavelength(stateI..., stateF...))
   ω0 = c_s / wavelength_transition
   ω = c_s / wavelength_tweezer


   I0 = U0 * 2 * ϵ0 * c_s / real(α)


   Beam = BeamProperties(Beam.Loc, Beam.dir, Beam.pol, waist, zr)
   BeamGaussian = GaussianBeam(Beam, I0, ω)
   return Tweezer(trapR, trapZ, BeamGaussian, ω0, I0, mass, α)
end






function Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapDepth; α::Number)
   mass = Atom.atom.mass
  


   trapR = (4*trapDepth/(mass*Beam.waist^2))^(1/2)
   trapZ =  (2*trapDepth/(mass*Beam.zr^2))^(1/2)


   wavelength_transition = abs(Atom.atom.getTransitionWavelength(stateI..., stateF...))
   ω0 = c_s / wavelength_transition
   ω = c_s / wavelength_tweezer


   I0 = trapDepth * 2 * ϵ0 * c_s/ real(α) # ASK FOR HELP HERE


   BeamGaussian = GaussianBeam(Beam, I0, ω)


   return Tweezer(trapR, trapZ, BeamGaussian, ω0, I0, mass, α)
end




Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR::Number, trapZ::Number) =  Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ, α =  get_groundStatePolarizability(Atom, stateI, stateF, wavelength_tweezer))
Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer::Number, trapDepth::Number) = Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapDepth, α = get_groundStatePolarizability(Atom, stateI, stateF, wavelength_tweezer))




#Multiple Polarizabilities
Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer::Number, trapDepth::Number, α_vec::Vector{T}) where T  = [Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapDepth, α =  α_i) for α_i in α_vec]
Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR::Number, trapZ::Number, α_vec::Vector{T})  where T=  [Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ, α = α_i) for α_i in α_vec]




function get_groundStatePolarizability(Atom::atomInterface, stateI, stateF, wavelength_tweezer)
   Damping_rate = Atom.atom.getTransitionRate(stateI..., stateF...)
   wavelength_transition = abs(Atom.atom.getTransitionWavelength(stateI..., stateF...))
   ω0 = c_s / wavelength_transition
   ω = c_s / wavelength_tweezer
   return 6 * pi * ϵ0 * c_s^3 * (Damping_rate / ω0^2) / (ω0^2 - ω^2 - 1im * (ω^3 / ω0^2) * Damping_rate)
end



function Tweezer_WF(BeamT::Tweezer, env::Environment_T, dl=1e-11)
   grid = env.grid
   PointGrid = ([i for i in Iterators.product(grid...)])
   Intensity = [get_Intensity(BeamT.Beam, [PointGrid[i, j, k]...]) for i in 1:size(PointGrid, 1), j in 1:size(PointGrid, 2), k in 1:size(PointGrid, 3)]
   push!(env.Intensity, Intensity)


   itp = interpolate(grid, Intensity, Gridded(Linear()))
   #ForceInterp(x, y, z) = 1 / (2 * C["e0"] * C["c"]) * real(BeamT.α) * [(itp[x-dl, y, z] + itp[x+dl, y, z]) / (2 * dl), (itp[x, y-dl, z] + itp[x, y+dl, z]) / (2 * dl), (itp[x, y, z-dl] + itp[x, y, z+dl]) / (2 * dl)]
   Ix, Iy, Iz = np_gradient(Intensity)
   Ix = extrapolate(interpolate(grid, Ix, Gridded(Linear())), 0)
   Iy = extrapolate(interpolate(grid, Iy, Gridded(Linear())), 0)
   Iz = extrapolate(interpolate(grid, Iz, Gridded(Linear())), 0)
   println(Ix[0, 0, 0], Iy[0, 0, 0], Iz[0, 0, 0])
   ForceInterp(x, y, z) = 1 / (2 * ϵ0 * c_s*step(grid[1]))* real(BeamT.α) * [Ix[x, y, z], Iy[x, y, z], Iz[x, y, z]]
  
   return Tweezer_WF(BeamT.trapR, BeamT.trapZ, BeamT.Beam, BeamT.mass, ForceInterp, BeamT.α)
end




function Tweezer_WF(BeamT::Vector{T}, env::Environment_T, dl=1e-11) where {T <: AbstractTweezer}
   grid = env.grid
   PointGrid = ([i for i in Iterators.product(grid...)])
   Intensity = [get_Intensity(BeamT[1].Beam, [PointGrid[i, j, k]...]) for i in 1:size(PointGrid, 1), j in 1:size(PointGrid, 2), k in 1:size(PointGrid, 3) ]
   push!(env.Intensity, Intensity)


   itp = interpolate(grid, Intensity, Gridded(Linear()))
   Ix, Iy, Iz = np_gradient(Intensity)
   Ix = extrapolate(interpolate(grid, Ix, Gridded(Linear())), 0)
   Iy = extrapolate(interpolate(grid, Iy, Gridded(Linear())), 0)
   Iz = extrapolate(interpolate(grid, Iz, Gridded(Linear())), 0)


   ForceInterp(x, y, z) = [1 / (2 * ϵ0 * c_s*step(grid[1]))* real(state.α) * [Ix[x, y, z], Iy[x, y, z], Iz[x, y, z]] for state in BeamT]
   α_vec = [state.α for state in BeamT]
   return Tweezer_WF(BeamT[1].trapR, BeamT[1].trapZ, BeamT[1].Beam, BeamT[1].mass, ForceInterp, α_vec)
end


function generateAtoms(TBeam::T, Temp, atomNum) where {T<:AbstractTweezer}
   coordList = []


   sigmaR = 1 / TBeam.trapR * sqrt(kb * Temp / TBeam.mass)
   sigmaZ = 1 / TBeam.trapZ * sqrt(kb * Temp / TBeam.mass)
   coordZ = rand(Normal(0, sigmaZ), atomNum)
   coordR = rand(Normal(0, sigmaR), atomNum)
   Beam_Dir = TBeam.Beam.beamStruct.dir
   Beam_Pos = TBeam.Beam.beamStruct.Loc
   for i in 1:atomNum
       x, y = coordR[i] * getSphereVec(2)
       z = coordZ[i]
       coord = [x, y, z]
       if !(Beam_Dir == [0, 0, 1])
           Rot = rotation_matrix(cross([0, 0, 1], Beam_Dir), get_angle([0, 0, 1], Beam_Dir))
           coord = Rot * coord
       end


       push!(coordList, coord + Beam_Pos)
   end
   return coordList
end


function sampleVelocities_BD(TBeam::T, Temp, atomNum) where {T<:AbstractTweezer} #check
   scale = sqrt(kb * Temp / TBeam.mass)
   vx, vy, vz = randn(atomNum) * scale, randn(atomNum) * scale, randn(atomNum) * scale
   velList = []
   for i in 1:atomNum
       push!(velList, [vx[i], vy[i], vz[i]])
   end
   return velList
end






function get_Fnet(TBeam::Tweezer_WF{T}, pos, vel) where {T<: Number}
   println(TBeam.ForceInterp(pos...))
   return TBeam.ForceInterp(pos...)
end


function get_Fnet(TBeam::Tweezer_WF{T}, pos, vel, rand::Any) where {T <: Number}
   return TBeam.ForceInterp(pos...)
end
get_Fnet_S(TBeam::Tweezer_WF{T}, pos, vel, rand::Any) where {T <: Number}= vcat(get_Fnet(TBeam::Tweezer_WF, pos, vel, rand::Any), [0.0])


function get_Fnet(TBeam::Tweezer_WF{Vector{T}}, pos, vel, state::Any) where {T <: Number}
   return TBeam.ForceInterp(pos...)[state + 1]
end




end
###
module SystemSetup
const c_s = 2.99798458E8
const ħ = 1.054571E-34
const kb = 1.3806452E-23
const ϵ0 =  8.85418E-12
const q_c = 1.60217663e-19
const m_e = 9.11e-31


using ..CoolTrap
using ..atom_class
using ..BeamClass
export System, set_tweezer, set_MOT, set_tweezer, join_Beams, clear_beams, set_SystemRHS, set_SystemRHS_SE, InitializeProblem, set_SystemRHS_MC, threadCount, clear_beams, clear_Tweezer, clear_MOT, join_beams
using StaticArrays
using StatsBase
using Random
using LinearAlgebra
mutable struct System
   AtomType
   TweezerConfig
   MOTConfig
   BeamConfig
   Environment
end


mutable struct atomInfo
   state::Int64
   last_em
end




function System(Atom::atomInterface, magneticField, gridBounds)
   return System(Atom, [], [], [], Environment_T([], magneticField, (gridBounds, gridBounds, gridBounds)))
end
function problemTypes(Sys::System)
   return ["Fabs_A", "Fabs_spont_R", "Fspont_R"]
end


function threadCount(Sys::System)
   return Threads.nthreads()
end


function InitializeProblem(Sys::System, atomNum,temp, problemType = "Fabs_A"; opt_args=nothing, default_tweezer = 1)
   if length(Sys.TweezerConfig) == 0
       return "Failed: No TweezerP"
   end
   TBeam = Sys.TweezerConfig[default_tweezer]
   atomPos = generateAtoms(TBeam, temp, atomNum)
   vel = sampleVelocities_BD(TBeam, temp, atomNum)
  
   u0 = []
   param = (opt_args, atomNum)
   atomInitialize = []


   if problemType == "Fabs_spont_R"
       for i in 1:atomNum
           push!(u0, vcat(atomPos[i]..., vel[i]..., 0)) #Last 0 is for number of photons scattered
           push!(atomInitialize, atomInfo(0, 0.0))           
       end
       param = (param..., atomInitialize)
       RHS = set_SystemRHS_MC(Sys, Int(length(u0)/atomNum))
   elseif problemType == "Fabs_A"
       for i in 1:atomNum
           push!(u0, vcat(atomPos[i]..., vel[i]..., 0))
       end
       RHS = set_SystemRHS(Sys, Int(length(u0)/atomNum))
   elseif problemType == "Fspont_R"
       for i in 1:atomNum
           push!(u0, vcat(atomPos[i]..., vel[i]..., 0))
           push!(atomInitialize, atomInfo(0, 0.0))           
       end
       param = (param..., atomInitialize)
       RHS = set_SystemRHS_SE(Sys, Int(length(u0)/atomNum))
   end
   return vcat(u0...), param, RHS
end




function set_tweezer(Sys::System, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ)
   gridbounds = Sys.Environment.grid[1]
   spacing = gridbounds[2] - gridbounds[1]
   TweezerP = Tweezer(Sys.AtomType, Beam, stateI, stateF, wavelength_tweezer, trapR, trapZ)
   TweezerA = Tweezer_WF(TweezerP, Sys.Environment, spacing )
   push!(Sys.TweezerConfig, TweezerA)
   #push!(Sys.BeamConfig, TweezerA)
end
function set_tweezer(Sys::System, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapDepth)
    gridbounds = Sys.Environment.grid[1]
    spacing = gridbounds[2] - gridbounds[1]
    TweezerP = Tweezer(Sys.AtomType, Beam, stateI, stateF, wavelength_tweezer, trapDepth)
    TweezerA = Tweezer_WF(TweezerP, Sys.Environment, spacing)
    push!(Sys.TweezerConfig, TweezerA)
end



function set_tweezer(Sys::System, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ; α)
    gridbounds = Sys.Environment.grid[1]
    spacing = gridbounds[2] - gridbounds[1]
    TweezerP = Tweezer(Sys.AtomType, Beam, stateI, stateF, wavelength_tweezer, trapR, trapZ; α =  α)
    TweezerA = Tweezer_WF(TweezerP, Sys.Environment, spacing)
    push!(Sys.TweezerConfig, TweezerA)
    #push!(Sys.BeamConfig, TweezerA)
 end
function set_tweezer(Sys::System, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapDepth; α)
    gridbounds = Sys.Environment.grid[1]
    spacing = gridbounds[2] - gridbounds[1]
    TweezerP = Tweezer(Sys.AtomType, Beam, stateI, stateF, wavelength_tweezer, trapDepth; α = α)
    TweezerA = Tweezer_WF(TweezerP, Sys.Environment, spacing)
    push!(Sys.TweezerConfig, TweezerA)
end



function set_MOT(Sys::System, Beam::BeamProperties, stateI, stateF, detuning, I, update=true)
   MOT_i = MOT_Beam(Sys.AtomType, Beam, Sys.Environment, stateI, stateF, detuning, I, update)
   push!(Sys.MOTConfig, MOT_i)
   #push!(Sys.BeamConfig, MOT_i)
end


function join_beams(Sys::System)
   Sys.BeamConfig = hcat(Sys.MOTConfig..., Sys.TweezerConfig...)
end
function clear_MOT(Sys::System)
   Sys.MOTConfig = []
end
function clear_Tweezer(Sys::System)
   Sys.TweezerConfig = []
end




function clear_beams(Sys::System)
   Sys.BeamConfig = []
end


function set_SystemRHS(Sys::System, ulength) #Handles absorption in a time averaged manner : NO SPONTANEOUS EMISSION
   function RHS(dy, y, p, t)
       AtomNum = p[2]
      
       for atom in 1:AtomNum
           offset::Int64 = ulength*(atom-1)
           dy[(4 + offset):(6 + offset)] = zeros(Float64, 3)
           for beam in Sys.BeamConfig


               dy[(4 + offset):(6 + offset)]  .+= get_Fnet(beam, y[(1 + offset):(3+ offset)], y[(4 + offset):(6 + offset)], p[1]) / Sys.AtomType.atom.mass# Fnet_i
           end
           dy[(1 + offset):(3+ offset)] = y[(4 + offset):(6 + offset)]
       end


   end
   return RHS
end
function set_SystemRHS_SE(Sys::System, ulength)  #Handles absorption in a time averaged manner : Spontaneous Emission randomly
   function RHS(dy, y, p, t)
       AtomNum, AtomInfo = p[end-1], p[end]
       @inbounds for atom in 1:AtomNum
           offset::Int64 = ulength*(atom-1)
           dy[(1 + offset):(3+ offset)] = y[(4 + offset):(6 + offset)]
           dy[(4 + offset):(6 + offset)] = zeros(Float64, 3)
           for beam in Sys.BeamConfig
               dy[(4 + offset):(6 + offset)]  .+= get_Fnet(beam, y[(1 + offset):(3+ offset)], y[(4 + offset):(6 + offset)], p[1]) / Sys.AtomType.atom.mass# Fnet_i
           end
           #spontMom = get_Spont(Sys, y[(1 + offset):(3+ offset)], t, AtomInfo[atom]);
           y[(4 + offset):(6 + offset)] .+=   get_Spont(Sys, t, AtomInfo[atom])/Sys.AtomType.atom.mass  #Sum over what exactly?
       end
   end
   return RHS
end


function get_Abs(Sys, pos, vel,  t, AtomInfo)
   dt = t-AtomInfo.last_em
   prob_req = [dt*get_scatteringrate_abs(beam, pos, vel) for beam in Sys.MOTConfig]
   prob_curr = rand(length(Sys.MOTConfig))
   indsOI = findall(prob_req .> prob_curr)
   if length(indsOI) == 0
       return zeros(Float64, 3)
   else
       indRun = sample(1:length(prob_req), Weights(prob_req./sum(prob_req)))


       AtomInfo.last_em = t
       AtomInfo.state = 1
       return get_momAbs(Sys.MOTConfig[indRun])#get_Fnet(Sys.MOTConfig[indRun], pos, vel)/ Sys.AtomType.atom.mass # get_momAbs(Sys.MOTConfig[indRun])
   end
end


function get_Spont(Sys, t, AtomInfo)
   dt = t-AtomInfo.last_em
   prob_req = [dt*beam.Γ_spont for beam in Sys.MOTConfig]
   prob_curr = rand(length(Sys.MOTConfig))
   indsOI = findall(prob_req .> prob_curr)
   if length(indsOI) == 0
       return zeros(Float64, 3)
   else
       indRun = argmax(prob_req[indsOI])


       AtomInfo.last_em = t
       AtomInfo.state = 0
       return get_momSpon(Sys.MOTConfig[indRun])
   end
end




function set_SystemRHS_MC(Sys::System, ulength) #Handles absorption and spontaneous emission in an MC fashion
   invMass = 1/Sys.AtomType.atom.mass
   function RHS(dy, y, p, t)
       AtomNum, AtomInfo = p[2], p[3]
       for atom in 1:AtomNum
           offset = ulength*(atom-1)
           dy[(1 + offset):(3 + offset)] = y[(4 + offset):(6 + offset)]
           dy[(4 + offset):(6 + offset)] = zeros(Float64, 3)


           if AtomInfo[atom].state == 0 #|| AtomInfo[atom].state == 1
               #dy[(4 + offset):(6 + offset)] = get_Abs(Sys, y[(1 + offset):(3 + offset)], y[(4 + offset):(6 + offset)],t, AtomInfo[atom])# Fnet_i
               y[(4 + offset):(6 + offset)] .+=  get_Abs(Sys, y[(1 + offset):(3 + offset)], y[(4 + offset):(6 + offset)],t, AtomInfo[atom])*invMass## Momentum approach
           else
               y[(4 + offset):(6 + offset)] .+=   get_Spont(Sys, t, AtomInfo[atom])*invMass#/Sys.AtomType.atom.mass#get_Spont(Sys, y[1*atom:3*atom], y[4*atom:6*atom], p[1])# Fnet_i
               if AtomInfo[atom].state == 0
                   y[(7 + offset)] = 1
               end
           end
           for beam in Sys.TweezerConfig
               dy[(4 + offset):(6 + offset)]  .+= get_Fnet(beam, y[(1 + offset):(3+ offset)], y[(4 + offset):(6 + offset)], AtomInfo[atom].state)*invMass# Fnet_i
           end
       end
      
   end
   return RHS
end


end


module IntegrateODE




using ..CoolTrap
using Base.Threads
using ..atom_class
using ..BeamClass
using ..SystemSetup
using StaticArrays
using StatsBase
using Random
using LinearAlgebra
using DifferentialEquations


export simulate, integrate, processData
function simulate(Sys::System, atomNum, temperature, IntegratorType, tspan, dt)
   solutions = Vector{Any}(undef, atomNum)  # preallocate solutions array
   @threads for i in 1:atomNum
       u0, param, RHS = InitializeProblem(Sys, 1, temperature, IntegratorType, opt_args=0.0)
       sol = integrate(Sys, u0, param, RHS, tspan, dt)
       solutions[i] = sol  # assign solution to specific index
   end
   return solutions
end


function integrate(Sys, u0, param, RHS, tspan, dt)
   boundMin, boundMax = minimum(Sys.Environment.grid[1]), maximum(Sys.Environment.grid[1])
   conditionStop(y, t, integrator) = !any(boundMin + 1e-6 .< y[1:3]) || !any(y[1:3] .< boundMax - 1e-6)
   affect!(integrator) = terminate!(integrator)
   cb = DiscreteCallback(conditionStop, affect!)


   prob = ODEProblem(RHS, u0, tspan, param)
   sol = solve(prob, RK4(), dt=dt, adaptive=false, callback = cb)#, callback=cb, abstol=1e-3, reltol=1e-3)
   return [sol.t, sol.u]
end


function processData(solutions)
   timeEval = solutions[1][1]
   x, y, z, vx, vy, vz, photons = [], [], [], [], [], [], []
   for index in eachindex(solutions)
       u = hcat(solutions[index][2]...)
       push!(x, u[1, :]); push!(y, u[2, :]); push!(z, u[3, :]); push!(vx, u[4, :]);  push!(vy, u[5, :]); push!(vz, u[6, :]); push!(photons, cumsum(u[7, :]))
   end
   return timeEval, x, y, z, vx, vy, vz, photons
end




end

