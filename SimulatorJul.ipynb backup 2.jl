{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m    Building\u001b[22m\u001b[39m Conda ─→ `~/.julia/scratchspaces/44cfe95a-1eb2-52ea-b672-e2afdf69b78f/915ebe6f0e7302693bdd8eac985797dba1d25662/build.log`\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m    Building\u001b[22m\u001b[39m PyCall → `~/.julia/scratchspaces/44cfe95a-1eb2-52ea-b672-e2afdf69b78f/43d304ac6f0354755f1d60730ece8c499980f7ba/build.log`\n"
     ]
    }
   ],
   "source": [
    "ENV[\"PYTHON\"] = \"/Users/jluke/miniforge3/bin/python\"\n",
    "using Pkg;\n",
    "Pkg.build(\"PyCall\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m    Updating\u001b[22m\u001b[39m registry at `~/.julia/registries/General.toml`\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Resolving\u001b[22m\u001b[39m package versions...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m  No Changes\u001b[22m\u001b[39m to `~/.julia/environments/v1.8/Project.toml`\n",
      "\u001b[32m\u001b[1m  No Changes\u001b[22m\u001b[39m to `~/.julia/environments/v1.8/Manifest.toml`\n"
     ]
    }
   ],
   "source": [
    "import Pkg;\n",
    "Pkg.add(\"Distributions\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "using DifferentialEquations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Main.atom_class"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "module atom_class\n",
    "\n",
    "export cesiumAtom, Energy, getSaturationIntensity, getTransitionRate, atomInterface, getw0\n",
    "\n",
    "using PyCall\n",
    "np = pyimport(\"numpy\")\n",
    "arc = pyimport(\"arc\")\n",
    "\n",
    "struct atomInterface\n",
    "    atom\n",
    "end\n",
    "function findStates(Atom::atomInterface, States, F::T, mF::T)::Float64 where {T<:AbstractFloat}\n",
    "    ind = findall(States[2] .== F .&& States[3] .== mF)\n",
    "    if length(ind) != 1\n",
    "        return \"Error\"\n",
    "    end\n",
    "    Energy::Matrix{Float64} = States[1]\n",
    "    return Energy[ind][1]\n",
    "end\n",
    "\n",
    "function Energy(Atom::atomInterface, n::T, l::T, j::G, f::G, mF::G, B::G) where {T<:Integer,G<:AbstractFloat}\n",
    "    ev2Hz = 2.417989242E14\n",
    "    States = Atom.atom.breitRabi(n, l, j, np.array([B]))\n",
    "    return findStates(Atom, States, f, mF) + Atom.atom.getEnergy(n, l, j) * ev2Hz\n",
    "end\n",
    "function getw0(Atom::atomInterface, stateI::T, stateF::T, B) where {T}\n",
    "    return Energy(Atom, stateF..., B) - Energy(Atom, stateI..., B)\n",
    "end\n",
    "function getSaturationIntensity(Atom::atomInterface, stateI::T, stateF::T) where {T}\n",
    "    return Atom.atom.getSaturationIntensity(stateI..., stateF...)\n",
    "end\n",
    "function getTransitionRate(Atom::atomInterface, stateI::T, stateF::T) where {T}\n",
    "    return Atom.atom.getTransitionRate(stateF[1:3]..., stateI[1:3]...)\n",
    "end\n",
    "\n",
    "Energy(Atom::atomInterface, n::T, l::T, j, f::T, mF::T, B::T) where {T<:Integer} = Energy(Atom, n, l, float(j), float(f), float(mF), float(B))\n",
    "Energy(Atom::atomInterface, n::T, l::T, j::G, f::T, mF::T, B::G) where {G<:AbstractFloat,T<:Integer} = Energy(Atom, n, l, float(j), float(f), float(mF), float(B))\n",
    "\n",
    "cesium = arc.Cesium\n",
    "\n",
    "\n",
    "const cesium_a = cesium()\n",
    "\n",
    "const cesiumAtom = atomInterface(cesium_a)\n",
    "\n",
    "end\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Main.utils"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "module utils\n",
    "using LinearAlgebra\n",
    "using Random\n",
    "using Statistics\n",
    "using Distributions\n",
    "\n",
    "export pol_v, decompose_spherical, project_onto_vector, project_onto_plane, rotation_matrix, get_angle, getSphereVec\n",
    "\n",
    "const pol_v = Dict(\"sigma_p\" => -1 / sqrt(2) * ComplexF64[1, 1im, 0], \"sigma_m\" => 1 / sqrt(2) * ComplexF64[1, -1im, 0], \"pi\" => ComplexF64[0, 0, 1])\n",
    "\n",
    "\n",
    "function decompose_spherical(pol)\n",
    "    return ComplexF64[dot(conj(pol_v[\"pi\"]), pol), dot(conj(pol_v[\"sigma_p\"]), pol), dot(conj(pol_v[\"sigma_m\"]), pol)]\n",
    "end\n",
    "\n",
    "project_onto_vector(v, n) = dot(v, n / norm(n)) * (n / norm(n))\n",
    "project_onto_plane(x, n) = x - dot(x, n / norm(n)) * (n / norm(n))\n",
    "\n",
    "\n",
    "function rotation_matrix(axis, theta)\n",
    "    axis = axis / norm(axis)\n",
    "    a = cos(theta / 2)\n",
    "    b, c, d = -axis * sin(theta / 2)\n",
    "    return [a*a+b*b-c*c-d*d 2*(b*c-a*d) 2*(b*d+a*c)\n",
    "        2*(b*c+a*d) a*a+c*c-b*b-d*d 2*(c*d-a*b)\n",
    "        2*(b*d-a*c) 2*(c*d+a*b) a*a+d*d-b*b-c*c]\n",
    "end\n",
    "\n",
    "get_angle(v1, v2) = acos(dot(v1, v2) / (norm(v1) * norm(v2)))\n",
    "\n",
    "function getSphereVec(n)\n",
    "    uvec = rand(Normal(), n)\n",
    "    uvec = uvec / norm(uvec)\n",
    "    return uvec\n",
    "end\n",
    "\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: replacing module BeamClass.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Main.BeamClass"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "module BeamClass\n",
    "using ..utils\n",
    "using LinearAlgebra\n",
    "export BeamProperties, GaussianBeam, get_Intensity\n",
    "\n",
    "struct BeamProperties{T}\n",
    "    Loc::T\n",
    "    dir::T\n",
    "    pol\n",
    "    waist::Float64\n",
    "    zr::Float64\n",
    "end\n",
    "\n",
    "BeamProperties(Loc::T, dir::T, pol) where {T} = BeamProperties{T}(Loc, dir, pol, 1e10, 1e10)\n",
    "struct GaussianBeam\n",
    "    beamStruct::BeamProperties\n",
    "    I0::Float64\n",
    "    waist::Float64\n",
    "end\n",
    "\n",
    "function get_Intensity(Beam::GaussianBeam, pos)\n",
    "    BP = Beam.beamStruct\n",
    "    r = norm(project_onto_plane(pos, BP.dir))\n",
    "    z = norm(project_onto_vector(pos, BP.dir) - BP.Loc)\n",
    "    w(z_i) = BP.waist * sqrt(1 + (z_i / BP.zr)^2)\n",
    "    I = Beam.I0 * (BP.waist / w(z))^2 * exp(-2 * r^2 / (w(r)^2))\n",
    "    return I\n",
    "end\n",
    "\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Main.Consts"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "module Consts\n",
    "export C\n",
    "const C = Dict(\"c\" => 2.99798458E8, \"hbar\" => 1.054571E-34, \"kb\" => 1.3806452E-23, \"e0\" => 8.85418E-12, \"q_c\" => 1.60217663e-19, \"m_e\" => 9.11e-31)\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Main.CoolTrap"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "module CoolTrap\n",
    "using ..BeamClass\n",
    "using ..utils\n",
    "using Interpolations\n",
    "\n",
    "using LinearAlgebra\n",
    "using Statistics: mean, std\n",
    "using Distributions\n",
    "\n",
    "using ..atom_class\n",
    "using ..Consts\n",
    "\n",
    "export MOT_Beam, get_Fnet, Tweezer, Tweezer_WF, generateAtoms, sampleVelocities_BD, Environment_T\n",
    "\n",
    "struct Environment_T\n",
    "    Intensity\n",
    "    B_Field\n",
    "    grid\n",
    "end\n",
    "\n",
    "# MOT FUNCTIONS\n",
    "struct MOT_Beam{T}\n",
    "    ω0::T\n",
    "    detuning::T\n",
    "    k_vec::Vector{T}\n",
    "    Γ::T\n",
    "    Isat::T\n",
    "    Beam::GaussianBeam\n",
    "    s0\n",
    "    environment\n",
    "    dir\n",
    "end\n",
    "function MOT_Beam(Atom::atomInterface, Beam::BeamProperties, environment, stateI, stateF, detuning, I, updateS0=true)\n",
    "    B_Field_func = environment.B_Field\n",
    "    ω0 = 2 * pi * getw0(Atom, stateI, stateF, norm(B_Field_func(0, 0, 0)))\n",
    "    BeamGaussian = GaussianBeam(Beam, I, ω0 + detuning)\n",
    "\n",
    "    wavelength = C[\"c\"] / (ω0 / (2 * pi))\n",
    "    k_vec = 2 * pi / wavelength * Beam.dir\n",
    "    Γ = getTransitionRate(Atom, stateI, stateF)\n",
    "    Isat = getSaturationIntensity(Atom, stateI, stateF)\n",
    "    q = stateF[end] - stateI[end]\n",
    "\n",
    "    rot_mat(pos) = rotation_matrix(cross(B_Field_func(pos...), Beam.dir), get_angle(B_Field_func(pos...), Beam.dir))\n",
    "    rotated_pol(pos) = rot_mat(pos) * Beam.pol\n",
    "    decomposed_pol(pos) = decompose_spherical(rotated_pol(pos))\n",
    "    s0 = pos -> 0\n",
    "\n",
    "    if !updateS0\n",
    "        println(\"Fixing\")\n",
    "        setpoint = get_Intensity(BeamGaussian, [0.0, 0.0, 0.0]) / Isat * abs(decomposed_pol([0.0, 0.0, 0.0])[q >= 0 ? q + 1 : end + 1 + q])^2 / norm(decomposed_pol([0.0, 0.0, 0.0]))^2\n",
    "        s0 = pos -> setpoint\n",
    "    else\n",
    "        println(\"Dynamic\")\n",
    "        s0 = pos -> get_Intensity(BeamGaussian, pos) / Isat * abs(decomposed_pol(pos)[q >= 0 ? q + 1 : end + 1 + q])^2 / norm(decomposed_pol(pos))^2\n",
    "    end\n",
    "\n",
    "    return MOT_Beam(ω0, detuning, k_vec, Γ, Isat, BeamGaussian, s0, environment, Beam.dir)\n",
    "end\n",
    "function get_scatteringrate_abs(MBeam::MOT_Beam, pos, vel)\n",
    "    return MBeam.s0(pos) * MBeam.Γ / 2 * 1 / (1 + MBeam.s0(pos) + (2 * (MBeam.detuning + dot(MBeam.k_vec, vel)) / MBeam.Γ)^2)\n",
    "end\n",
    "function get_scatteringrate(MBeam::MOT_Beam, pos)\n",
    "    return MBeam.s0(pos) * MBeam.Γ / 2 * 1 / (1 + MBeam.s0(pos) + (2 * MBeam.detuning / MBeam.Γ)^2)\n",
    "end\n",
    "function get_Fabs(MBeam::MOT_Beam, pos, vel)\n",
    "    return C[\"hbar\"] * (MBeam.ω0 + MBeam.detuning) / C[\"c\"] * get_scatteringrate_abs(MBeam, pos, vel) * MBeam.dir\n",
    "end\n",
    "function get_Fspon(MBeam::MOT_Beam, pos)\n",
    "    return C[\"hbar\"] * norm(MBeam.k_vec) * get_scatteringrate(MBeam, pos) * getSphereVec(3)\n",
    "end\n",
    "function get_Fnet(MBeam::MOT_Beam, pos, vel)\n",
    "    Spont = get_Fspon(MBeam, pos)\n",
    "    Abs = get_Fabs(MBeam, pos, vel)\n",
    "    return Abs + Spont\n",
    "end\n",
    "\n",
    "abstract type AbstractTweezer end\n",
    "\n",
    "# Tweezer Function \n",
    "struct Tweezer{T} <: AbstractTweezer\n",
    "    trapR::T\n",
    "    trapZ::T\n",
    "    Beam::GaussianBeam\n",
    "    ω0::T\n",
    "    I0::T\n",
    "    mass::T\n",
    "    α\n",
    "end\n",
    "\n",
    "struct Tweezer_WF{T} <: AbstractTweezer\n",
    "    trapR::T\n",
    "    trapZ::T\n",
    "    Beam::GaussianBeam\n",
    "    mass::T\n",
    "    ForceInterp\n",
    "    α\n",
    "end\n",
    "\n",
    "function np_gradient(f::AbstractArray{T,N}; h=1.0) where {T,N}\n",
    "    g = ntuple(N) do dim\n",
    "        grad = similar(f)\n",
    "        ax = axes(f, dim)\n",
    "        # forward difference on the left boundary\n",
    "        I1 = first(ax)\n",
    "        I2 = I1 + 1\n",
    "        for I in CartesianIndices(axes(f))\n",
    "            if I[dim] < size(f, dim)\n",
    "                grad[I] = (f[I + CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))] - f[I]) / h\n",
    "            end\n",
    "        end\n",
    "        # central difference in the interior points\n",
    "        for i in ax[2:end-1]\n",
    "            I1 = i - 1\n",
    "            I2 = i + 1\n",
    "            for I in CartesianIndices(axes(f))\n",
    "                if I[dim] > 1 && I[dim] < size(f, dim)\n",
    "                    grad[I] = (f[I + CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))] - f[I - CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))]) / (2h)\n",
    "                end\n",
    "            end\n",
    "        end\n",
    "        # backward difference on the right boundary\n",
    "        I1 = last(ax) - 1\n",
    "        I2 = last(ax)\n",
    "        for I in CartesianIndices(axes(f))\n",
    "            if I[dim] > 1\n",
    "                grad[I] = (f[I] - f[I - CartesianIndex(ntuple(d -> d == dim ? 1 : 0, N))]) / h\n",
    "            end\n",
    "        end\n",
    "        grad\n",
    "    end\n",
    "    g\n",
    "end\n",
    "\n",
    "function Tweezer(Atom::atomInterface, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ)\n",
    "    mass = Atom.atom.mass\n",
    "\n",
    "    waist = trapR / trapZ * (1 / (pi * sqrt(2))) * wavelength_tweezer\n",
    "    zr = pi * waist^2 / wavelength_tweezer\n",
    "    U0 = 1 / 4 * mass * waist^2 * trapR^2\n",
    "    Damping_rate = Atom.atom.getTransitionRate(stateI..., stateF...)\n",
    "    wavelength_transition = abs(Atom.atom.getTransitionWavelength(stateI..., stateF...))\n",
    "    ω0 = C[\"c\"] / wavelength_transition\n",
    "    ω = C[\"c\"] / wavelength_tweezer\n",
    "\n",
    "    α = 6 * pi * C[\"e0\"] * C[\"c\"]^3 * (Damping_rate / ω0^2) / (ω0^2 - ω^2 - 1im * (ω^3 / ω0^2) * Damping_rate)\n",
    "    I0 = U0 * 2 * C[\"e0\"] * C[\"c\"] / real(α)\n",
    "\n",
    "    Beam = BeamProperties(Beam.Loc, Beam.dir, Beam.pol, waist, zr)\n",
    "    BeamGaussian = GaussianBeam(Beam, I0, ω)\n",
    "\n",
    "    return Tweezer(trapR, trapZ, BeamGaussian, ω0, I0, mass, α)\n",
    "end\n",
    "function Tweezer_WF(BeamT::Tweezer, env::Environment_T, dl=1e-11)\n",
    "    grid = env.grid\n",
    "    PointGrid = ([i for i in Iterators.product(grid...)])\n",
    "    Intensity = [get_Intensity(BeamT.Beam, [PointGrid[i, j, k]...]) for i in 1:size(PointGrid, 1), j in 1:size(PointGrid, 2), k in 1:size(PointGrid, 3)]\n",
    "    push!(env.Intensity, Intensity)\n",
    "\n",
    "    itp = interpolate(grid, Intensity, Gridded(Linear()))\n",
    "    #ForceInterp(x, y, z) = 1 / (2 * C[\"e0\"] * C[\"c\"]) * real(BeamT.α) * [(itp[x-dl, y, z] + itp[x+dl, y, z]) / (2 * dl), (itp[x, y-dl, z] + itp[x, y+dl, z]) / (2 * dl), (itp[x, y, z-dl] + itp[x, y, z+dl]) / (2 * dl)]\n",
    "    Ix, Iy, Iz = np_gradient(Intensity)\n",
    "    Ix = interpolate(grid, Ix, Gridded(Linear()))\n",
    "    Iy = interpolate(grid, Iy, Gridded(Linear()))\n",
    "    Iz = interpolate(grid, Iz, Gridded(Linear()))\n",
    "    ForceInterp(x, y, z) = 1 / (2 * C[\"e0\"] * C[\"c\"]*step(grid[1]))* real(BeamT.α) * [Ix[x, y, z], Iy[x, y, z], Iz[x, y, z]]\n",
    "    return Tweezer_WF(BeamT.trapR, BeamT.trapZ, BeamT.Beam, BeamT.mass, ForceInterp, BeamT.α)\n",
    "end\n",
    "\n",
    "\n",
    "function generateAtoms(TBeam::T, Temp, atomNum) where {T<:AbstractTweezer}\n",
    "    coordList = []\n",
    "    sigmaR = 1 / TBeam.trapR * sqrt(C[\"kb\"] * T / TBeam.mass)\n",
    "    sigmaZ = 1 / TBeam.trapZ * sqrt(C[\"kb\"] * T / TBeam.mass)\n",
    "    coordZ = rand(Normal(0, sigmaZ), AtomNum)\n",
    "    coordR = rand(Normal(0, sigmaR), AtomNum)\n",
    "    Beam_Dir = TBeam.Beam.beamStruct.dir\n",
    "    Beam_Pos = TBeam.Beam.beamStruct.Loc\n",
    "    for i in 1:AtomNum\n",
    "        x, y = coordR[i] * getSphereVec(2)\n",
    "        z = coordZ[i]\n",
    "        coord = [x, y, z]\n",
    "        if !(Beam_Dir == [0, 0, 1])\n",
    "            Rot = rotation_matrix(cross([0, 0, 1], Beam_Dir), get_angle([0, 0, 1], Beam_Dir))\n",
    "            coord = Rot * coord\n",
    "        end\n",
    "\n",
    "        push!(coordList, coord + Beam_Pos)\n",
    "    end\n",
    "    return coordList\n",
    "end\n",
    "\n",
    "\n",
    "function sampleVelocities_BD(TBeam::T, Temp, atomNum) where {T<:AbstractTweezer} #check \n",
    "    scale = sqrt(C[\"kb\"] * T / TBeam.mass)\n",
    "    vx, vy, vz = randn(num_particles) * a_Rb, randn(num_particles) * a_Rb, randn(num_particles) * a_Rb\n",
    "    velList = []\n",
    "    for i in 1:atomNum\n",
    "        push!(velList, [vx[i], vy[i], vz[i]])\n",
    "    end\n",
    "    return velList\n",
    "end\n",
    "\n",
    "\n",
    "function get_Fnet(TBeam::Tweezer_WF, pos, vel)\n",
    "    return TBeam.ForceInterp(pos...)\n",
    "end\n",
    "\n",
    "\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Main.SystemSetup"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "module SystemSetup\n",
    "\n",
    "using ..CoolTrap\n",
    "using ..atom_class\n",
    "using ..BeamClass\n",
    "\n",
    "export System, set_tweezer, set_MOT, set_tweezer, join_Beams, clear_beams, set_SystemRHS\n",
    "\n",
    "mutable struct System\n",
    "    AtomType\n",
    "    TweezerConfig\n",
    "    MOTConfig\n",
    "    BeamConfig\n",
    "    Environment\n",
    "end\n",
    "\n",
    "function System(Atom::atomInterface, magneticField, gridBounds)\n",
    "    return System(Atom, [], [], [], Environment_T([], magneticField, (gridBounds, gridBounds, gridBounds)))\n",
    "end\n",
    "\n",
    "function set_tweezer(Sys::System, Beam::BeamProperties, stateI, stateF, wavelength_tweezer, trapR, trapZ)\n",
    "    gridbounds = Sys.Environment.grid[1]\n",
    "    spacing = gridbounds[2] - gridbounds[1]\n",
    "    TweezerP = Tweezer(Sys.AtomType, Beam, stateI, stateF, wavelength_tweezer, trapR, trapZ)\n",
    "    TweezerA = Tweezer_WF(TweezerP, Sys.Environment, spacing )\n",
    "    push!(Sys.TweezerConfig, TweezerA)\n",
    "    push!(Sys.BeamConfig, TweezerA)\n",
    "end\n",
    "function set_MOT(Sys::System, Beam::BeamProperties, stateI, stateF, detuning, I, update=true)\n",
    "    MOT_i = MOT_Beam(Sys.AtomType, Beam, Sys.Environment, stateI, stateF, detuning, I, update)\n",
    "    push!(Sys.MOTConfig, MOT_i)\n",
    "    push!(Sys.BeamConfig, MOT_i)\n",
    "end\n",
    "\n",
    "\n",
    "function clear_beams(Sys::System)\n",
    "    System.TweezerConfig = []\n",
    "    System.MOTConfig = []\n",
    "    System.BeamConfig = []\n",
    "end\n",
    "\n",
    "function set_SystemRHS(Sys::System)\n",
    "    function RHS(dy, y, p, t)\n",
    "        #pos_vec = y[1:3]\n",
    "        #vel_vec = y[4:6]\n",
    "        dy[1:3] = y[4:6]\n",
    "        dy[4:6] = zeros(3)\n",
    "        for beam in Sys.BeamConfig\n",
    "            #Fnet_i = get_Fnet(beam, y[1:3], y[4:6])\n",
    "            dy[4:6] .+= get_Fnet(beam, y[1:3], y[4:6])# Fnet_i\n",
    "        end\n",
    "        dy[4:6] = dy[4:6] / Sys.AtomType.atom.mass\n",
    "    end\n",
    "    return RHS\n",
    "end\n",
    "\n",
    "\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "using ProfileView\n",
    "using BenchmarkTools"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "ename": "LoadError",
     "evalue": "LoadError: UndefVarError: @btime not defined\nin expression starting at /Users/jluke/Desktop/Simulation/HighFieldImaging/SimulatorJul.ipynb:1",
     "output_type": "error",
     "traceback": [
      "LoadError: UndefVarError: @btime not defined\n",
      "in expression starting at /Users/jluke/Desktop/Simulation/HighFieldImaging/SimulatorJul.ipynb:1\n"
     ]
    }
   ],
   "source": [
    "@btime get_Fnet(OurSystem.MOTConfig[1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Setup of System"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(-1.5e-5, 1.5e-5)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "boundMin, boundMax = minimum(OurSystem.Environment.grid[1]), maximum(OurSystem.Environment.grid[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Setting up MOT\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fixing\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fixing\n",
      "Setting up Tweezer\n"
     ]
    }
   ],
   "source": [
    "using .atom_class\n",
    "using .SystemSetup\n",
    "using .utils\n",
    "using .BeamClass\n",
    "using .CoolTrap\n",
    "\n",
    "function createSystem()\n",
    "    BField(x, y, z) = 800E-4 * [0, 0, 1]\n",
    "    atomType = cesiumAtom\n",
    "    OurSystem = System(atomType, BField, -4e-6:0.4e-7:4e-6)\n",
    "    return OurSystem\n",
    "end\n",
    "function createLight!(OurSystem)\n",
    "    #clear_beams(OurSystem)\n",
    "    stateF = (6, 1, 1.5, 5, 5)\n",
    "    stateI = (6, 0, 0.5, 4, 4)\n",
    "    detuning = 3e6\n",
    "    I_mot = 14\n",
    "    beamAngle = 45 * pi / 180\n",
    "\n",
    "\n",
    "    println(\"Setting up MOT\")\n",
    "    updates0 = false\n",
    "    LocAll::Vector{Float64} = [0, 0, 0]\n",
    "    set_MOT(OurSystem, BeamProperties(LocAll, [-1.0, 0.0, 0.0], pol_v[\"sigma_m\"]),  stateI, stateF, detuning, I_mot, updates0)\n",
    "    set_MOT(OurSystem, BeamProperties(LocAll, [1.0, 0.0, 0.0], pol_v[\"sigma_p\"]),  stateI, stateF, detuning, I_mot, updates0)\n",
    "    #R1\n",
    "    #set_MOT(OurSystem, BeamProperties(LocAll, [0, sin(beamAngle), cos(beamAngle)], pol_v[\"sigma_m\"]),  stateI, stateF, detuning, I_mot,updates0)\n",
    "    #set_MOT(OurSystem, BeamProperties(LocAll, [0, -sin(beamAngle), -cos(beamAngle)], pol_v[\"sigma_p\"]),  stateI, stateF, detuning, I_mot, updates0)\n",
    "    #R2\n",
    "    #set_MOT(OurSystem, BeamProperties(LocAll, [0, -sin(beamAngle), cos(beamAngle)], pol_v[\"sigma_m\"]),  stateI, stateF, detuning, I_mot, updates0)\n",
    "    #set_MOT(OurSystem, BeamProperties(LocAll, [0, sin(beamAngle), -cos(beamAngle)], pol_v[\"sigma_p\"]),  stateI, stateF, detuning, I_mot, updates0)\n",
    "\n",
    "    println(\"Setting up Tweezer\")\n",
    "\n",
    "    #Tweezer\n",
    "    stateI_T = (6, 1, 1 / 2)\n",
    "    stateF_T = (6, 0, 1 / 2)\n",
    "    wavelength_tweezer = 1064E-9\n",
    "    trapR, trapZ = 2 * pi * 0.150e6, 2 * pi * 0.029e6\n",
    "\n",
    "    set_tweezer(OurSystem, BeamProperties(LocAll, [0.0, 0.0, 1.0], pol_v[\"pi\"]), stateI_T, stateF_T, wavelength_tweezer, trapR, trapZ)\n",
    "    #join_Beams(OurSystem)\n",
    "end\n",
    "\n",
    "OurSystem = createSystem()\n",
    "createLight!(OurSystem);\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "simulate (generic function with 1 method)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "function simulate(Sys::System)\n",
    "    RHS = set_SystemRHS(Sys)\n",
    "\n",
    "    boundMin, boundMax = minimum(OurSystem.Environment.grid[1]), maximum(OurSystem.Environment.grid[1])\n",
    "    conditionStop(y, t, integrator) = !any(boundMin + 1e-6 .< y[1:3] .< boundMax - 1e-6)\n",
    "    affect!(integrator) = terminate!(integrator)\n",
    "    cb = DiscreteCallback(conditionStop, affect!)\n",
    "\n",
    "\n",
    "\n",
    "    u0 = zeros(Float64, 6)\n",
    "    tspan = (0, 10e-3)\n",
    "    prob = ODEProblem(RHS, u0, tspan)\n",
    "    dt = 1e-9\n",
    "    sol = solve(prob, Euler(), dt=dt, callback=cb)\n",
    "    return sol.t, hcat(sol.u...)\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "t, u = simulate(OurSystem)\n",
    "x, y, z, vx, vy, vz = u[1, :],  u[2, :],  u[3, :],  u[4, :], u[5, :], u[6, :];"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000002,)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PyObject <module 'matplotlib.pyplot' from '/Users/jluke/miniforge3/lib/python3.8/site-packages/matplotlib/pyplot.py'>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "using PyCall\n",
    "plt = pyimport(\"matplotlib.pyplot\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(t, x)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "using DifferentialEquations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "true"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "min_, max_ = 1, 3\n",
    "d = [1, 2, 3]\n",
    "any(min_ .< d .< max_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.0, 1.0], [1.1, 1.1], [1.2100000000000002, 1.2100000000000002], [1.3310000000000002, 1.3310000000000002], [1.4641000000000002, 1.4641000000000002], [1.61051, 1.61051], [1.7715610000000002, 1.7715610000000002], [1.9487171, 1.9487171], [2.1435888100000002, 2.1435888100000002], [2.357947691, 2.357947691], [2.5937424601, 2.5937424601], [2.8531167061100002, 2.8531167061100002], [3.1384283767210004, 3.1384283767210004], [3.4522712143931003, 3.4522712143931003], [3.7974983358324104, 3.7974983358324104], [4.177248169415652, 4.177248169415652], [4.594972986357217, 4.594972986357217], [5.054470284992938, 5.054470284992938], [5.559917313492232, 5.559917313492232], [6.115909044841455, 6.115909044841455], [6.727499949325598, 6.727499949325598]]\n"
     ]
    }
   ],
   "source": [
    "using DifferentialEquations\n",
    "\n",
    "function f(du, u, p, t)\n",
    "    du[1] = u[1]\n",
    "    du[2] = u[2]\n",
    "end\n",
    "\n",
    "threshold = 10\n",
    "u0 = [1.0, 1.0]\n",
    "tspan = (0.0, 2.0)\n",
    "prob = ODEProblem(f, u0, tspan)\n",
    "\n",
    "condition(u, t, integrator) = u[1] > threshold\n",
    "affect!(integrator) = terminate!(integrator)      # How to respond to the event\n",
    "cb = DiscreteCallback(condition, affect!)\n",
    "\n",
    "dt = 0.1\n",
    "sol = solve(prob, Euler(), dt=dt, callback=cb)\n",
    "\n",
    "println(sol.u)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.4840294736005524"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "using Interpolations\n",
    "\n",
    "# Let's assume you have a 3D grid `x`, `y`, `z`, and corresponding `intensities`\n",
    "# Here I just generate some random data\n",
    "x = 1:1000\n",
    "y = 1:1000\n",
    "z = 1:1000\n",
    "intensities = rand(1000, 1000, 1000)  # replace this with your actual data\n",
    "\n",
    "# Create a grid of points in 3D space\n",
    "grid = (x, y, z)\n",
    "\n",
    "# Create the interpolation object\n",
    "itp = interpolate(grid, intensities, Gridded(Linear()))\n",
    "\n",
    "# Now you can get interpolated values at any point within the grid:\n",
    "val = itp[1.5, 2.5, 3.5]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "300×300×300 Array{Int64, 3}:\n",
       "[:, :, 1] =\n",
       "   3    3    3    3    3    3    3    3  …    3    3    3    3    3    3    3\n",
       "   3    3    3    3    3    3    3    3       3    3    3    3    3    3    3\n",
       "   5    5    5    5    5    5    5    5       5    5    5    5    5    5    5\n",
       "   7    7    7    7    7    7    7    7       7    7    7    7    7    7    7\n",
       "   9    9    9    9    9    9    9    9       9    9    9    9    9    9    9\n",
       "  11   11   11   11   11   11   11   11  …   11   11   11   11   11   11   11\n",
       "  13   13   13   13   13   13   13   13      13   13   13   13   13   13   13\n",
       "  15   15   15   15   15   15   15   15      15   15   15   15   15   15   15\n",
       "  17   17   17   17   17   17   17   17      17   17   17   17   17   17   17\n",
       "  19   19   19   19   19   19   19   19      19   19   19   19   19   19   19\n",
       "   ⋮                        ⋮            ⋱              ⋮                 \n",
       " 583  583  583  583  583  583  583  583     583  583  583  583  583  583  583\n",
       " 585  585  585  585  585  585  585  585     585  585  585  585  585  585  585\n",
       " 587  587  587  587  587  587  587  587     587  587  587  587  587  587  587\n",
       " 589  589  589  589  589  589  589  589     589  589  589  589  589  589  589\n",
       " 591  591  591  591  591  591  591  591  …  591  591  591  591  591  591  591\n",
       " 593  593  593  593  593  593  593  593     593  593  593  593  593  593  593\n",
       " 595  595  595  595  595  595  595  595     595  595  595  595  595  595  595\n",
       " 597  597  597  597  597  597  597  597     597  597  597  597  597  597  597\n",
       " 599  599  599  599  599  599  599  599     599  599  599  599  599  599  599\n",
       "\n",
       "[:, :, 2] =\n",
       "   3    3    3    3    3    3    3    3  …    3    3    3    3    3    3    3\n",
       "   3    3    3    3    3    3    3    3       3    3    3    3    3    3    3\n",
       "   5    5    5    5    5    5    5    5       5    5    5    5    5    5    5\n",
       "   7    7    7    7    7    7    7    7       7    7    7    7    7    7    7\n",
       "   9    9    9    9    9    9    9    9       9    9    9    9    9    9    9\n",
       "  11   11   11   11   11   11   11   11  …   11   11   11   11   11   11   11\n",
       "  13   13   13   13   13   13   13   13      13   13   13   13   13   13   13\n",
       "  15   15   15   15   15   15   15   15      15   15   15   15   15   15   15\n",
       "  17   17   17   17   17   17   17   17      17   17   17   17   17   17   17\n",
       "  19   19   19   19   19   19   19   19      19   19   19   19   19   19   19\n",
       "   ⋮                        ⋮            ⋱              ⋮                 \n",
       " 583  583  583  583  583  583  583  583     583  583  583  583  583  583  583\n",
       " 585  585  585  585  585  585  585  585     585  585  585  585  585  585  585\n",
       " 587  587  587  587  587  587  587  587     587  587  587  587  587  587  587\n",
       " 589  589  589  589  589  589  589  589     589  589  589  589  589  589  589\n",
       " 591  591  591  591  591  591  591  591  …  591  591  591  591  591  591  591\n",
       " 593  593  593  593  593  593  593  593     593  593  593  593  593  593  593\n",
       " 595  595  595  595  595  595  595  595     595  595  595  595  595  595  595\n",
       " 597  597  597  597  597  597  597  597     597  597  597  597  597  597  597\n",
       " 599  599  599  599  599  599  599  599     599  599  599  599  599  599  599\n",
       "\n",
       "[:, :, 3] =\n",
       "   3    3    3    3    3    3    3    3  …    3    3    3    3    3    3    3\n",
       "   3    3    3    3    3    3    3    3       3    3    3    3    3    3    3\n",
       "   5    5    5    5    5    5    5    5       5    5    5    5    5    5    5\n",
       "   7    7    7    7    7    7    7    7       7    7    7    7    7    7    7\n",
       "   9    9    9    9    9    9    9    9       9    9    9    9    9    9    9\n",
       "  11   11   11   11   11   11   11   11  …   11   11   11   11   11   11   11\n",
       "  13   13   13   13   13   13   13   13      13   13   13   13   13   13   13\n",
       "  15   15   15   15   15   15   15   15      15   15   15   15   15   15   15\n",
       "  17   17   17   17   17   17   17   17      17   17   17   17   17   17   17\n",
       "  19   19   19   19   19   19   19   19      19   19   19   19   19   19   19\n",
       "   ⋮                        ⋮            ⋱              ⋮                 \n",
       " 583  583  583  583  583  583  583  583     583  583  583  583  583  583  583\n",
       " 585  585  585  585  585  585  585  585     585  585  585  585  585  585  585\n",
       " 587  587  587  587  587  587  587  587     587  587  587  587  587  587  587\n",
       " 589  589  589  589  589  589  589  589     589  589  589  589  589  589  589\n",
       " 591  591  591  591  591  591  591  591  …  591  591  591  591  591  591  591\n",
       " 593  593  593  593  593  593  593  593     593  593  593  593  593  593  593\n",
       " 595  595  595  595  595  595  595  595     595  595  595  595  595  595  595\n",
       " 597  597  597  597  597  597  597  597     597  597  597  597  597  597  597\n",
       " 599  599  599  599  599  599  599  599     599  599  599  599  599  599  599\n",
       "\n",
       ";;; … \n",
       "\n",
       "[:, :, 298] =\n",
       "   3    3    3    3    3    3    3    3  …    3    3    3    3    3    3    3\n",
       "   3    3    3    3    3    3    3    3       3    3    3    3    3    3    3\n",
       "   5    5    5    5    5    5    5    5       5    5    5    5    5    5    5\n",
       "   7    7    7    7    7    7    7    7       7    7    7    7    7    7    7\n",
       "   9    9    9    9    9    9    9    9       9    9    9    9    9    9    9\n",
       "  11   11   11   11   11   11   11   11  …   11   11   11   11   11   11   11\n",
       "  13   13   13   13   13   13   13   13      13   13   13   13   13   13   13\n",
       "  15   15   15   15   15   15   15   15      15   15   15   15   15   15   15\n",
       "  17   17   17   17   17   17   17   17      17   17   17   17   17   17   17\n",
       "  19   19   19   19   19   19   19   19      19   19   19   19   19   19   19\n",
       "   ⋮                        ⋮            ⋱              ⋮                 \n",
       " 583  583  583  583  583  583  583  583     583  583  583  583  583  583  583\n",
       " 585  585  585  585  585  585  585  585     585  585  585  585  585  585  585\n",
       " 587  587  587  587  587  587  587  587     587  587  587  587  587  587  587\n",
       " 589  589  589  589  589  589  589  589     589  589  589  589  589  589  589\n",
       " 591  591  591  591  591  591  591  591  …  591  591  591  591  591  591  591\n",
       " 593  593  593  593  593  593  593  593     593  593  593  593  593  593  593\n",
       " 595  595  595  595  595  595  595  595     595  595  595  595  595  595  595\n",
       " 597  597  597  597  597  597  597  597     597  597  597  597  597  597  597\n",
       " 599  599  599  599  599  599  599  599     599  599  599  599  599  599  599\n",
       "\n",
       "[:, :, 299] =\n",
       "   3    3    3    3    3    3    3    3  …    3    3    3    3    3    3    3\n",
       "   3    3    3    3    3    3    3    3       3    3    3    3    3    3    3\n",
       "   5    5    5    5    5    5    5    5       5    5    5    5    5    5    5\n",
       "   7    7    7    7    7    7    7    7       7    7    7    7    7    7    7\n",
       "   9    9    9    9    9    9    9    9       9    9    9    9    9    9    9\n",
       "  11   11   11   11   11   11   11   11  …   11   11   11   11   11   11   11\n",
       "  13   13   13   13   13   13   13   13      13   13   13   13   13   13   13\n",
       "  15   15   15   15   15   15   15   15      15   15   15   15   15   15   15\n",
       "  17   17   17   17   17   17   17   17      17   17   17   17   17   17   17\n",
       "  19   19   19   19   19   19   19   19      19   19   19   19   19   19   19\n",
       "   ⋮                        ⋮            ⋱              ⋮                 \n",
       " 583  583  583  583  583  583  583  583     583  583  583  583  583  583  583\n",
       " 585  585  585  585  585  585  585  585     585  585  585  585  585  585  585\n",
       " 587  587  587  587  587  587  587  587     587  587  587  587  587  587  587\n",
       " 589  589  589  589  589  589  589  589     589  589  589  589  589  589  589\n",
       " 591  591  591  591  591  591  591  591  …  591  591  591  591  591  591  591\n",
       " 593  593  593  593  593  593  593  593     593  593  593  593  593  593  593\n",
       " 595  595  595  595  595  595  595  595     595  595  595  595  595  595  595\n",
       " 597  597  597  597  597  597  597  597     597  597  597  597  597  597  597\n",
       " 599  599  599  599  599  599  599  599     599  599  599  599  599  599  599\n",
       "\n",
       "[:, :, 300] =\n",
       "   3    3    3    3    3    3    3    3  …    3    3    3    3    3    3    3\n",
       "   3    3    3    3    3    3    3    3       3    3    3    3    3    3    3\n",
       "   5    5    5    5    5    5    5    5       5    5    5    5    5    5    5\n",
       "   7    7    7    7    7    7    7    7       7    7    7    7    7    7    7\n",
       "   9    9    9    9    9    9    9    9       9    9    9    9    9    9    9\n",
       "  11   11   11   11   11   11   11   11  …   11   11   11   11   11   11   11\n",
       "  13   13   13   13   13   13   13   13      13   13   13   13   13   13   13\n",
       "  15   15   15   15   15   15   15   15      15   15   15   15   15   15   15\n",
       "  17   17   17   17   17   17   17   17      17   17   17   17   17   17   17\n",
       "  19   19   19   19   19   19   19   19      19   19   19   19   19   19   19\n",
       "   ⋮                        ⋮            ⋱              ⋮                 \n",
       " 583  583  583  583  583  583  583  583     583  583  583  583  583  583  583\n",
       " 585  585  585  585  585  585  585  585     585  585  585  585  585  585  585\n",
       " 587  587  587  587  587  587  587  587     587  587  587  587  587  587  587\n",
       " 589  589  589  589  589  589  589  589     589  589  589  589  589  589  589\n",
       " 591  591  591  591  591  591  591  591  …  591  591  591  591  591  591  591\n",
       " 593  593  593  593  593  593  593  593     593  593  593  593  593  593  593\n",
       " 595  595  595  595  595  595  595  595     595  595  595  595  595  595  595\n",
       " 597  597  597  597  597  597  597  597     597  597  597  597  597  597  597\n",
       " 599  599  599  599  599  599  599  599     599  599  599  599  599  599  599"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "checker{typeof(f)}"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "struct checker{T}\n",
    "    a\n",
    "    b::T\n",
    "end\n",
    "f(x) = 1\n",
    "typeof(checker(1, f))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3-element Vector{Float64}:\n",
       " 7.725442287185756e6\n",
       " 7.725439176555662e6\n",
       " 7.725439739985274e6"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "dl = 1e-7\n",
    "F(x, y, z) = 5 * [(itp[x-dl, y, z] + itp[x+dl, y, z]) / (2 * dl), (itp[x, y-dl, z] + itp[x, y+dl, z]) / (2 * dl), (itp[x, y, z-dl] + itp[x, y, z+dl]) / (2 * dl)]\n",
    "F(2, 2, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "struct Environment\n",
    "    Intensity\n",
    "    B_Field\n",
    "end\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Checker (generic function with 1 method)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "struct check\n",
    "    s\n",
    "end\n",
    "\n",
    "function Checker()\n",
    "    f(x) = x^2\n",
    "    s(x) = f(x) + 1\n",
    "    check(s)\n",
    "end\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.0e-8"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "1e-3 / 2e4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n",
      "4\n",
      "5\n"
     ]
    }
   ],
   "source": [
    "array = [1, 2, 3, 4, 5]\n",
    "println(array[end])  # Outputs: 5\n",
    "println(array[end-1])  # Outputs: 4\n",
    "python_index = -1 # Python index\n",
    "\n",
    "println(array[python_index >= 0 ? python_index + 1 : end + 1 + python_index])  # Prints the last element of the array\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.19996800255986344"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "using .utils\n",
    "dir = [0, 0, 1]\n",
    "loc = [0, 0, 0]\n",
    "waist = 100\n",
    "function test(pos, dir, Loc, waist)\n",
    "    zr = 1\n",
    "    r = norm(project_onto_plane(pos, dir))\n",
    "    z = norm(project_onto_plane(pos, dir) - Loc)\n",
    "    w(z_i) = waist * sqrt(1 + (z_i / zr)^2)\n",
    "    I = (waist / w(z))^2 * exp(-2 * r^2 / (w(r)^2))\n",
    "end\n",
    "x = 1:1:100\n",
    "y = 1:1:100\n",
    "z = 1:1:100\n",
    "test([0, 2, 1], dir, loc, waist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "100×100×100 interpolate((1:1:100,1:1:100,1:1:100), ::Array{Float64, 3}, Gridded(Linear())) with element type Float64:\n",
       "[:, :, 1] =\n",
       " 0.333289     0.166639     0.0908926    …  0.000101989  9.996e-5\n",
       " 0.166639     0.111091     0.0714153       0.000101958  9.993e-5\n",
       " 0.0908926    0.0714153    0.0526216       0.000101906  9.98801e-5\n",
       " 0.0555451    0.04761      0.0384541       0.000101833  9.98103e-5\n",
       " 0.0370299    0.0333269    0.0285659       0.00010174   9.97207e-5\n",
       " 0.0263107    0.0243855    0.0217349    …  0.000101626  9.96114e-5\n",
       " 0.019604     0.0185149    0.0169458       0.000101492  9.94826e-5\n",
       " 0.0151485    0.0144899    0.0135108       0.000101338  9.93343e-5\n",
       " 0.0120458    0.0116256    0.0109868       0.000101164  9.91668e-5\n",
       " 0.00980198   0.00952192   0.00908911      0.00010097   9.89803e-5\n",
       " ⋮                                      ⋱               \n",
       " 0.000118096  0.000118054  0.000117984     5.47356e-5   5.41457e-5\n",
       " 0.00011557   0.00011553   0.000115464     5.41868e-5   5.36086e-5\n",
       " 0.000113125  0.000113087  0.000113023     5.36431e-5   5.30764e-5\n",
       " 0.000110757  0.00011072   0.000110659     5.31046e-5   5.25491e-5\n",
       " 0.000108462  0.000108426  0.000108368  …  5.25712e-5   5.20269e-5\n",
       " 0.000106237  0.000106204  0.000106147     5.20431e-5   5.15095e-5\n",
       " 0.000104081  0.000104048  0.000103994     5.15201e-5   5.09972e-5\n",
       " 0.000101989  0.000101958  0.000101906     5.10024e-5   5.04899e-5\n",
       " 9.996e-5     9.993e-5     9.98801e-5      5.04899e-5   4.99875e-5\n",
       "\n",
       "[:, :, 2] =\n",
       " 0.333289     0.166639     0.0908926    …  0.000101989  9.996e-5\n",
       " 0.166639     0.111091     0.0714153       0.000101958  9.993e-5\n",
       " 0.0908926    0.0714153    0.0526216       0.000101906  9.98801e-5\n",
       " 0.0555451    0.04761      0.0384541       0.000101833  9.98103e-5\n",
       " 0.0370299    0.0333269    0.0285659       0.00010174   9.97207e-5\n",
       " 0.0263107    0.0243855    0.0217349    …  0.000101626  9.96114e-5\n",
       " 0.019604     0.0185149    0.0169458       0.000101492  9.94826e-5\n",
       " 0.0151485    0.0144899    0.0135108       0.000101338  9.93343e-5\n",
       " 0.0120458    0.0116256    0.0109868       0.000101164  9.91668e-5\n",
       " 0.00980198   0.00952192   0.00908911      0.00010097   9.89803e-5\n",
       " ⋮                                      ⋱               \n",
       " 0.000118096  0.000118054  0.000117984     5.47356e-5   5.41457e-5\n",
       " 0.00011557   0.00011553   0.000115464     5.41868e-5   5.36086e-5\n",
       " 0.000113125  0.000113087  0.000113023     5.36431e-5   5.30764e-5\n",
       " 0.000110757  0.00011072   0.000110659     5.31046e-5   5.25491e-5\n",
       " 0.000108462  0.000108426  0.000108368  …  5.25712e-5   5.20269e-5\n",
       " 0.000106237  0.000106204  0.000106147     5.20431e-5   5.15095e-5\n",
       " 0.000104081  0.000104048  0.000103994     5.15201e-5   5.09972e-5\n",
       " 0.000101989  0.000101958  0.000101906     5.10024e-5   5.04899e-5\n",
       " 9.996e-5     9.993e-5     9.98801e-5      5.04899e-5   4.99875e-5\n",
       "\n",
       "[:, :, 3] =\n",
       " 0.333289     0.166639     0.0908926    …  0.000101989  9.996e-5\n",
       " 0.166639     0.111091     0.0714153       0.000101958  9.993e-5\n",
       " 0.0908926    0.0714153    0.0526216       0.000101906  9.98801e-5\n",
       " 0.0555451    0.04761      0.0384541       0.000101833  9.98103e-5\n",
       " 0.0370299    0.0333269    0.0285659       0.00010174   9.97207e-5\n",
       " 0.0263107    0.0243855    0.0217349    …  0.000101626  9.96114e-5\n",
       " 0.019604     0.0185149    0.0169458       0.000101492  9.94826e-5\n",
       " 0.0151485    0.0144899    0.0135108       0.000101338  9.93343e-5\n",
       " 0.0120458    0.0116256    0.0109868       0.000101164  9.91668e-5\n",
       " 0.00980198   0.00952192   0.00908911      0.00010097   9.89803e-5\n",
       " ⋮                                      ⋱               \n",
       " 0.000118096  0.000118054  0.000117984     5.47356e-5   5.41457e-5\n",
       " 0.00011557   0.00011553   0.000115464     5.41868e-5   5.36086e-5\n",
       " 0.000113125  0.000113087  0.000113023     5.36431e-5   5.30764e-5\n",
       " 0.000110757  0.00011072   0.000110659     5.31046e-5   5.25491e-5\n",
       " 0.000108462  0.000108426  0.000108368  …  5.25712e-5   5.20269e-5\n",
       " 0.000106237  0.000106204  0.000106147     5.20431e-5   5.15095e-5\n",
       " 0.000104081  0.000104048  0.000103994     5.15201e-5   5.09972e-5\n",
       " 0.000101989  0.000101958  0.000101906     5.10024e-5   5.04899e-5\n",
       " 9.996e-5     9.993e-5     9.98801e-5      5.04899e-5   4.99875e-5\n",
       "\n",
       ";;; … \n",
       "\n",
       "[:, :, 98] =\n",
       " 0.333289     0.166639     0.0908926    …  0.000101989  9.996e-5\n",
       " 0.166639     0.111091     0.0714153       0.000101958  9.993e-5\n",
       " 0.0908926    0.0714153    0.0526216       0.000101906  9.98801e-5\n",
       " 0.0555451    0.04761      0.0384541       0.000101833  9.98103e-5\n",
       " 0.0370299    0.0333269    0.0285659       0.00010174   9.97207e-5\n",
       " 0.0263107    0.0243855    0.0217349    …  0.000101626  9.96114e-5\n",
       " 0.019604     0.0185149    0.0169458       0.000101492  9.94826e-5\n",
       " 0.0151485    0.0144899    0.0135108       0.000101338  9.93343e-5\n",
       " 0.0120458    0.0116256    0.0109868       0.000101164  9.91668e-5\n",
       " 0.00980198   0.00952192   0.00908911      0.00010097   9.89803e-5\n",
       " ⋮                                      ⋱               \n",
       " 0.000118096  0.000118054  0.000117984     5.47356e-5   5.41457e-5\n",
       " 0.00011557   0.00011553   0.000115464     5.41868e-5   5.36086e-5\n",
       " 0.000113125  0.000113087  0.000113023     5.36431e-5   5.30764e-5\n",
       " 0.000110757  0.00011072   0.000110659     5.31046e-5   5.25491e-5\n",
       " 0.000108462  0.000108426  0.000108368  …  5.25712e-5   5.20269e-5\n",
       " 0.000106237  0.000106204  0.000106147     5.20431e-5   5.15095e-5\n",
       " 0.000104081  0.000104048  0.000103994     5.15201e-5   5.09972e-5\n",
       " 0.000101989  0.000101958  0.000101906     5.10024e-5   5.04899e-5\n",
       " 9.996e-5     9.993e-5     9.98801e-5      5.04899e-5   4.99875e-5\n",
       "\n",
       "[:, :, 99] =\n",
       " 0.333289     0.166639     0.0908926    …  0.000101989  9.996e-5\n",
       " 0.166639     0.111091     0.0714153       0.000101958  9.993e-5\n",
       " 0.0908926    0.0714153    0.0526216       0.000101906  9.98801e-5\n",
       " 0.0555451    0.04761      0.0384541       0.000101833  9.98103e-5\n",
       " 0.0370299    0.0333269    0.0285659       0.00010174   9.97207e-5\n",
       " 0.0263107    0.0243855    0.0217349    …  0.000101626  9.96114e-5\n",
       " 0.019604     0.0185149    0.0169458       0.000101492  9.94826e-5\n",
       " 0.0151485    0.0144899    0.0135108       0.000101338  9.93343e-5\n",
       " 0.0120458    0.0116256    0.0109868       0.000101164  9.91668e-5\n",
       " 0.00980198   0.00952192   0.00908911      0.00010097   9.89803e-5\n",
       " ⋮                                      ⋱               \n",
       " 0.000118096  0.000118054  0.000117984     5.47356e-5   5.41457e-5\n",
       " 0.00011557   0.00011553   0.000115464     5.41868e-5   5.36086e-5\n",
       " 0.000113125  0.000113087  0.000113023     5.36431e-5   5.30764e-5\n",
       " 0.000110757  0.00011072   0.000110659     5.31046e-5   5.25491e-5\n",
       " 0.000108462  0.000108426  0.000108368  …  5.25712e-5   5.20269e-5\n",
       " 0.000106237  0.000106204  0.000106147     5.20431e-5   5.15095e-5\n",
       " 0.000104081  0.000104048  0.000103994     5.15201e-5   5.09972e-5\n",
       " 0.000101989  0.000101958  0.000101906     5.10024e-5   5.04899e-5\n",
       " 9.996e-5     9.993e-5     9.98801e-5      5.04899e-5   4.99875e-5\n",
       "\n",
       "[:, :, 100] =\n",
       " 0.333289     0.166639     0.0908926    …  0.000101989  9.996e-5\n",
       " 0.166639     0.111091     0.0714153       0.000101958  9.993e-5\n",
       " 0.0908926    0.0714153    0.0526216       0.000101906  9.98801e-5\n",
       " 0.0555451    0.04761      0.0384541       0.000101833  9.98103e-5\n",
       " 0.0370299    0.0333269    0.0285659       0.00010174   9.97207e-5\n",
       " 0.0263107    0.0243855    0.0217349    …  0.000101626  9.96114e-5\n",
       " 0.019604     0.0185149    0.0169458       0.000101492  9.94826e-5\n",
       " 0.0151485    0.0144899    0.0135108       0.000101338  9.93343e-5\n",
       " 0.0120458    0.0116256    0.0109868       0.000101164  9.91668e-5\n",
       " 0.00980198   0.00952192   0.00908911      0.00010097   9.89803e-5\n",
       " ⋮                                      ⋱               \n",
       " 0.000118096  0.000118054  0.000117984     5.47356e-5   5.41457e-5\n",
       " 0.00011557   0.00011553   0.000115464     5.41868e-5   5.36086e-5\n",
       " 0.000113125  0.000113087  0.000113023     5.36431e-5   5.30764e-5\n",
       " 0.000110757  0.00011072   0.000110659     5.31046e-5   5.25491e-5\n",
       " 0.000108462  0.000108426  0.000108368  …  5.25712e-5   5.20269e-5\n",
       " 0.000106237  0.000106204  0.000106147     5.20431e-5   5.15095e-5\n",
       " 0.000104081  0.000104048  0.000103994     5.15201e-5   5.09972e-5\n",
       " 0.000101989  0.000101958  0.000101906     5.10024e-5   5.04899e-5\n",
       " 9.996e-5     9.993e-5     9.98801e-5      5.04899e-5   4.99875e-5"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "grid = (x, y, z)\n",
    "grid_z = ([i for i in Iterators.product(grid...)])\n",
    "results = [test([grid_z[i, j, k]...], dir, loc, waist) for i in 1:size(grid_z, 1), j in 1:size(grid_z, 2), k in 1:size(grid_z, 3)]\n",
    "itp = interpolate(grid, results, Gridded(Linear()))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "111"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2131415161718191"
     ]
    }
   ],
   "source": [
    "for i in x\n",
    "    print(i)\n",
    "end\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "g.g(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3-element Vector{Int64}:\n",
       " -3\n",
       "  6\n",
       " -3"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "using LinearAlgebra\n",
    "cross([1, 2, 3], [4, 5, 6])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "using BenchmarkTools"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  4.202 μs (84 allocations: 7.44 KiB)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "retcode: Success\n",
       "Interpolation: 3rd order Hermite\n",
       "t: 11-element Vector{Float64}:\n",
       " 0.0\n",
       " 0.1\n",
       " 0.20875983490136413\n",
       " 0.35726389787661994\n",
       " 0.5257564990101543\n",
       " 0.7240176327413396\n",
       " 0.9429638825953713\n",
       " 1.1837165911596974\n",
       " 1.4414730997835639\n",
       " 1.7147710521468877\n",
       " 2.0\n",
       "u: 11-element Vector{Vector{Float64}}:\n",
       " [1.0, 1.0]\n",
       " [1.1051708333333334, 1.1051708333333334]\n",
       " [1.2321488065875676, 1.2321488065875676]\n",
       " [1.4294120036068518, 1.4294120036068518]\n",
       " [1.6917352728997825, 1.6917352728997825]\n",
       " [2.0626957828502146, 2.0626957828502146]\n",
       " [2.567561239545397, 2.567561239545397]\n",
       " [3.266449792380289, 3.266449792380289]\n",
       " [4.2268310840609615, 4.2268310840609615]\n",
       " [5.555233110989807, 5.555233110989807]\n",
       " [7.388737785739548, 7.388737785739548]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "function RHS_t!(du, u, p, t)\n",
    "    du[1] = u[1]\n",
    "    du[2] = u[2]\n",
    "end\n",
    "u0 = [1.0, 1.0]\n",
    "tspan = (0, 2.0)\n",
    "prob = ODEProblem(RHS_t, u0, tspan)\n",
    "dt = 0.1\n",
    "@btime sol = solve(prob, RK4(), dt=dt)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m    Updating\u001b[22m\u001b[39m registry at `~/.julia/registries/General.toml`\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Resolving\u001b[22m\u001b[39m package versions...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DomainSets ───────────────── v0.5.15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m GR_jll ───────────────────── v0.72.8+0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Sundials_jll ─────────────── v5.2.1+0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Bijections ───────────────── v0.1.4\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DiffEqPhysics ────────────── v3.11.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DifferentialEquations ────── v6.18.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m ThreadsX ─────────────────── v0.1.11\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m SteadyStateDiffEq ────────── v1.12.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m BoundaryValueDiffEq ──────── v2.8.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m JuliaFormatter ───────────── v0.21.2\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m AutoHashEquals ───────────── v0.2.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m MultivariatePolynomials ──── v0.3.18\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m CompositeTypes ───────────── v0.1.3\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m BandedMatrices ───────────── v0.16.8\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m StaticArraysCore ─────────── v1.4.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Symbolics ────────────────── v4.3.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m SafeTestsets ─────────────── v0.0.1\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m NaNMath ──────────────────── v0.3.7\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DimensionalPlotRecipes ───── v1.2.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m LeftChildRightSiblingTrees ─ v0.1.3\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m CodecZlib ────────────────── v0.7.2\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m CommonMark ───────────────── v0.8.12\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m NDTensors ────────────────── v0.2.2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DynamicPolynomials ───────── v0.3.21\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m ArrayLayouts ─────────────── v0.4.11\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m ParameterizedFunctions ───── v5.15.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DelayDiffEq ──────────────── v5.33.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Sundials ─────────────────── v4.15.1\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m ModelingToolkit ──────────── v8.1.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m FlameGraphs ──────────────── v0.2.10\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Tokenize ─────────────────── v0.5.25\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m AbstractTrees ────────────── v0.3.4\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m DiffEqFinancial ──────────── v2.4.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Crayons ──────────────────── v4.1.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m LineSearches ─────────────── v7.1.1\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m IntervalSets ─────────────── v0.7.3\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m GPUCompiler ──────────────── v0.21.4\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m MultiScaleArrays ─────────── v1.10.0\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Latexify ─────────────────── v0.15.21\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m TermInterface ────────────── v0.2.3\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Primes ───────────────────── v0.5.4\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m GR ───────────────────────── v0.72.8\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m TensorOperations ─────────── v0.7.1\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m CSTParser ────────────────── v3.3.6\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Metatheory ───────────────── v1.3.5\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m SymbolicUtils ────────────── v0.19.11\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m    Updating\u001b[22m\u001b[39m `~/.julia/environments/v1.9/Project.toml`\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [0c46a032] \u001b[39m\u001b[92m+ DifferentialEquations v6.18.0\u001b[39m\n",
      "\u001b[32m\u001b[1m    Updating\u001b[22m\u001b[39m `~/.julia/environments/v1.9/Manifest.toml`\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[33m⌅\u001b[39m\u001b[90m [1520ce14] \u001b[39m\u001b[95m↓ AbstractTrees v0.4.4 ⇒ v0.3.4\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [4c555306] \u001b[39m\u001b[92m+ ArrayLayouts v0.4.11\u001b[39m\n",
      " \u001b[90m [15f4f7f2] \u001b[39m\u001b[92m+ AutoHashEquals v0.2.0\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [aae01518] \u001b[39m\u001b[92m+ BandedMatrices v0.16.8\u001b[39m\n",
      " \u001b[90m [e2ed5e7c] \u001b[39m\u001b[92m+ Bijections v0.1.4\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [764a87c0] \u001b[39m\u001b[92m+ BoundaryValueDiffEq v2.8.0\u001b[39m\n",
      " \u001b[90m [00ebfdb7] \u001b[39m\u001b[92m+ CSTParser v3.3.6\u001b[39m\n",
      " \u001b[90m [944b1d66] \u001b[39m\u001b[93m↑ CodecZlib v0.7.1 ⇒ v0.7.2\u001b[39m\n",
      " \u001b[90m [a80b9123] \u001b[39m\u001b[92m+ CommonMark v0.8.12\u001b[39m\n",
      " \u001b[90m [b152e2b5] \u001b[39m\u001b[92m+ CompositeTypes v0.1.3\u001b[39m\n",
      " \u001b[90m [a8cc5b0e] \u001b[39m\u001b[92m+ Crayons v4.1.1\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [bcd4f6db] \u001b[39m\u001b[92m+ DelayDiffEq v5.33.0\u001b[39m\n",
      " \u001b[90m [5a0ffddc] \u001b[39m\u001b[92m+ DiffEqFinancial v2.4.0\u001b[39m\n",
      " \u001b[90m [055956cb] \u001b[39m\u001b[92m+ DiffEqPhysics v3.11.0\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [0c46a032] \u001b[39m\u001b[92m+ DifferentialEquations v6.18.0\u001b[39m\n",
      " \u001b[90m [c619ae07] \u001b[39m\u001b[92m+ DimensionalPlotRecipes v1.2.0\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [5b8099bc] \u001b[39m\u001b[92m+ DomainSets v0.5.15\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [7c1d4256] \u001b[39m\u001b[92m+ DynamicPolynomials v0.3.21\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [08572546] \u001b[39m\u001b[95m↓ FlameGraphs v1.0.0 ⇒ v0.2.10\u001b[39m\n",
      " \u001b[90m [61eb1bfa] \u001b[39m\u001b[93m↑ GPUCompiler v0.21.3 ⇒ v0.21.4\u001b[39m\n",
      " \u001b[90m [28b8d3ca] \u001b[39m\u001b[93m↑ GR v0.72.7 ⇒ v0.72.8\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [8197267c] \u001b[39m\u001b[95m↓ IntervalSets v0.7.4 ⇒ v0.7.3\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [98e50ef6] \u001b[39m\u001b[92m+ JuliaFormatter v0.21.2\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [23fbe1c1] \u001b[39m\u001b[95m↓ Latexify v0.16.1 ⇒ v0.15.21\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [1d6d02ad] \u001b[39m\u001b[95m↓ LeftChildRightSiblingTrees v0.2.0 ⇒ v0.1.3\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [d3d80556] \u001b[39m\u001b[95m↓ LineSearches v7.2.0 ⇒ v7.1.1\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [e9d8d322] \u001b[39m\u001b[92m+ Metatheory v1.3.5\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [961ee093] \u001b[39m\u001b[92m+ ModelingToolkit v8.1.0\u001b[39m\n",
      " \u001b[90m [f9640e96] \u001b[39m\u001b[92m+ MultiScaleArrays v1.10.0\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [102ac46a] \u001b[39m\u001b[92m+ MultivariatePolynomials v0.3.18\u001b[39m\n",
      " \u001b[90m [23ae76d9] \u001b[39m\u001b[93m↑ NDTensors v0.2.1 ⇒ v0.2.2\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [77ba4419] \u001b[39m\u001b[95m↓ NaNMath v1.0.2 ⇒ v0.3.7\u001b[39m\n",
      " \u001b[90m [65888b18] \u001b[39m\u001b[92m+ ParameterizedFunctions v5.15.0\u001b[39m\n",
      " \u001b[90m [27ebfcd6] \u001b[39m\u001b[93m↑ Primes v0.5.3 ⇒ v0.5.4\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [1bc83da4] \u001b[39m\u001b[92m+ SafeTestsets v0.0.1\u001b[39m\n",
      " \u001b[90m [1e83bf80] \u001b[39m\u001b[93m↑ StaticArraysCore v1.4.0 ⇒ v1.4.1\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [9672c7b4] \u001b[39m\u001b[92m+ SteadyStateDiffEq v1.12.0\u001b[39m\n",
      " \u001b[90m [5e0ebb24] \u001b[39m\u001b[93m↑ Strided v1.2.3 ⇒ v2.0.1\u001b[39m\n",
      " \u001b[90m [4db3bf67] \u001b[39m\u001b[92m+ StridedViews v0.1.2\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [c3572dad] \u001b[39m\u001b[92m+ Sundials v4.15.1\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [d1185830] \u001b[39m\u001b[92m+ SymbolicUtils v0.19.11\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [0c5d862f] \u001b[39m\u001b[92m+ Symbolics v4.3.0\u001b[39m\n",
      "\u001b[32m⌃\u001b[39m\u001b[90m [6aa20fa7] \u001b[39m\u001b[95m↓ TensorOperations v1.3.1 ⇒ v0.7.1\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [8ea1fca8] \u001b[39m\u001b[92m+ TermInterface v0.2.3\u001b[39m\n",
      " \u001b[90m [ac1d9e8a] \u001b[39m\u001b[92m+ ThreadsX v0.1.11\u001b[39m\n",
      " \u001b[90m [0796e94c] \u001b[39m\u001b[92m+ Tokenize v0.5.25\u001b[39m\n",
      " \u001b[90m [d2c73de3] \u001b[39m\u001b[93m↑ GR_jll v0.72.7+0 ⇒ v0.72.8+0\u001b[39m\n",
      "\u001b[33m⌅\u001b[39m\u001b[90m [fb77eaff] \u001b[39m\u001b[92m+ Sundials_jll v5.2.1+0\u001b[39m\n",
      " \u001b[90m [8bb1440f] \u001b[39m\u001b[93m~ DelimitedFiles ⇒ \u001b[39m\n",
      " \u001b[90m [05823500] \u001b[39m\u001b[91m- OpenLibm_jll v0.8.1+0\u001b[39m\n",
      " \u001b[90m [bea87d4a] \u001b[39m\u001b[92m+ SuiteSparse_jll v5.10.1+0\u001b[39m\n",
      "\u001b[36m\u001b[1m        Info\u001b[22m\u001b[39m Packages marked with \u001b[32m⌃\u001b[39m and \u001b[33m⌅\u001b[39m have new versions available, but those with \u001b[33m⌅\u001b[39m are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1mPrecompiling\u001b[22m\u001b[39m "
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "project...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSafeTestsets\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mAutoHashEquals\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mTermInterface\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mNaNMath\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mCompositeTypes\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mBijections\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mIntervalSets\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mAbstractTrees\u001b[39m\n",
      "\u001b[33m  ✓ \u001b[39m\u001b[90mStaticArraysCore\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mCodecZlib\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mPrimes\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mCrayons\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDimensionalPlotRecipes\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSundials_jll\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mTokenize\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGR_jll\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGraphics\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mLeftChildRightSiblingTrees\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffRules\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mTensorOperations\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mMethodAnalysis\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mLatexify\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mThreadsX\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffResults\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mMultivariatePolynomials\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mArrayLayouts\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mMAT\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mWignerSymbols\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mCairo\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mRecipesPipeline\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mCSTParser\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mUnitfulLatexify\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGR\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGPUCompiler\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mFlameGraphs\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDynamicPolynomials\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[33m  ✓ \u001b[39mStaticArrays\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mMetatheory\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mCommonMark\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mFiniteDiff\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mLuxurySparse\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mBandedMatrices\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mLabelledArrays\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mResettableStacks\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mArnoldiMethod\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mRecursiveArrayTools\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[33m  ✓ \u001b[39mInterpolations\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDomainSets\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mKernelAbstractions\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGtk\u001b[39m\n",
      "\u001b[32m  ✓ \u001b[39m\u001b[90mForwardDiff\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mNDTensors\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mJuliaFormatter\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSIMDDualNumbers\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGraphs\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mLightGraphs\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mNLSolversBase\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mGtkObservables\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSciMLBase\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mVertexSafeGraphs\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mLineSearches\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mMetal\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSparseDiffTools\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mNLsolve\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mProfileView\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSymbolicUtils\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mOptim\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mITensors\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mLoopVectorization\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mPastaQ\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mMathOptInterface\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mRecursiveFactorization\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[91m  ✗ \u001b[39mSCS\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[91m  ✗ \u001b[39mQuantumInformation\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mNonlinearSolve\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSymbolics\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffEqBase\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mPlots\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffEqCallbacks\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffEqJump\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mBoundaryValueDiffEq\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffEqNoiseProcess\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffEqPhysics\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[91m  ✗ \u001b[39m\u001b[90mSteadyStateDiffEq\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDiffEqFinancial\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mSundials\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mModelingToolkit\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mOrdinaryDiffEq\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mParameterizedFunctions\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mDelayDiffEq\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mStochasticDiffEq\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39m\u001b[90mMultiScaleArrays\u001b[39m\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m  ✓ \u001b[39mQuantumOptics\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[91m  ✗ \u001b[39mDifferentialEquations\n",
      "  90 dependencies successfully precompiled in 80 seconds. 319 already precompiled. 6 skipped during auto due to previous errors.\n",
      "  \u001b[33m3\u001b[39m dependencies precompiled but different versions are currently loaded. Restart julia to access the new versions\n",
      "  \u001b[91m4\u001b[39m dependencies errored. To see a full report either run `import Pkg; Pkg.precompile()` or load the packages\n"
     ]
    }
   ],
   "source": [
    "import Pkg;\n",
    "Pkg.add(\"DifferentialEquations\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "BeamProperties{Vector{Int64}}([1, 2, 3], [3, 4, 5], [6, 7, 8], 1000.0, 10000.0)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "using .BeamClass\n",
    "\n",
    "loc = [1, 2, 3]\n",
    "dir = [3, 4, 5]\n",
    "pol = [6, 7, 8]\n",
    "BeamProperties(loc, dir, pol)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3-element Vector{Float64}:\n",
       " 1.0\n",
       " 2.0\n",
       " 3.0"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "a = [1.0, 2.0, 3.0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "asdf\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "BeamProperties{Vector{Int64}}([1, 2, 3], [3, 4, 5], [6, 7, 8], 1.0e10, 1.0e10)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2-element Vector{Float64}:\n",
       " -0.9546875109947802\n",
       "  0.2976100743432445"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: redefinition of constant Cesium. This may fail, cause incorrect answers, or produce other errors.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "2.206946951453701e-25"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "const Cesium = cesium()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.206946951453701e-25"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "check (generic function with 2 methods)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "stateF = (6, 1, 1.5, 5, 5)\n",
    "stateI = (6, 0, 0.5, 4, 4)\n",
    "function check(a, b, c, d)\n",
    "    return a + b + c\n",
    "end\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "13"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "check(stateF[1:2]..., stateI[1:2]...)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.17989242e22"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "417989242E14"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "check (generic function with 1 method)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Julia 1.8.2",
   "language": "julia",
   "name": "julia-1.8"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.8.2"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
