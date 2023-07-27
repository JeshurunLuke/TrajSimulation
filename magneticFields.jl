### IOFFE + QUADRAPOLE 

module quadrapole

const u0 = pi*4E-7
const amu = 1.66E-27
const mRb87 = 87*1.67E-27
const g = 9.81
const c = 299792458
const kb = 1.38066E-23
const uB = 9.27E-24
const uRb87 = uB
const mz = 1
const hbar = 6.626E-34/(2*pi)
const itom = 0.0254
const J2Ghz = 1.5092E+24 
const BtoU = 0.7

export BfieldMag, Bnorm
using SpecialFunctions
using LinearAlgebra
abstract type AbstractField end

mutable struct BfieldMag <: AbstractField
    Iq0::Float64
    Ii0::Float64
    mbound::Float64
    TurnsQ::Float64
    Sq::Float64
    Rq0::Float64
    dq0::Float64
    imbound::Float64
    Iturns::Float64
    Si::Float64
    Ri0::Float64
    di0::Float64
    delx::Float64
    dely::Float64
end

#Constructor
BfieldMag(Iq0::Float64, Ii0::Float64) = BfieldMag(Iq0, Ii0, 0.25*itom, 131.0, 0.006*itom, (1.625*itom + 131.0*0.006*itom)/2, 1.45/2*itom+ 0.25*itom, 0.25*itom/2, 19.0, 0.006*itom, (0.297*itom + 19.0*0.006*itom)/2, (0.409*0.0254 + 0.25*itom/2), 0, 0)

function kq(B::BfieldMag, radial, Radius, Zdist)
    return sqrt((4*Radius*radial)/((Radius+radial)^2+Zdist^2))
end

function getVar(B::BfieldMag, x, y, z, n, m, dq)
    radial = sqrt(x^2 + y^2)
    Radius = B.Rq0 + n*B.Sq
    Zdist = z-(dq+m)
    return radial, Radius, Zdist
end

function kqI(B::BfieldMag, radial, Radius, Zdist)
    return sqrt((4*Radius*radial)/((Radius+radial)^2+Zdist^2))
end

function getVari(B::BfieldMag, x,y,z,n,m,dq)
    radial = sqrt((y-B.dely)^2+z^2)
    Radius = B.Ri0 + n*B.Si
    Zdist = x-(dq+B.delx+m*B.Si)
    return radial, Radius, Zdist
end

function Bnorm(B::BfieldMag, x,y,z)
    if B.Ii0 == 0 && x == 0 && y == 0 && z == 0 
        return [0 ,0, 0]
    end

    coord = [x,y,z]
    Bqz, Bqy, Bqx, Biz, Biy, Bix = 0, 0 , 0, 0, 0, 0
    m = 0 
    for n in -Int((B.TurnsQ-1)/2):Int(((B.TurnsQ-1)/2))
        Bqz +=  bqz(B,n, m, B.dq0, coord) - bqz(B,n, m, - B.dq0, coord)
        Bqy +=  bqy(B,n, m,  B.dq0, coord) - bqy(B,n, m, - B.dq0, coord)
        Bqx +=  bqx(B,n, m,  B.dq0, coord) - bqx(B,n, m, - B.dq0, coord)
    end
    if B.Ii0 != 0
        for n in -Int((B.Iturns-1)/2):Int((B.Iturns-1)/2)
            Biz += sign(z)*biz(B,n, m, B.di0, coord)
            Biy += sign(y)*biy(B,n, m, B.di0, coord)
            Bix += bix(B,n, m, B.di0, coord)
        end
    end
    Bqx = sign(x)*Bqx 
    Bqy = sign(y)*Bqy
    Biy = sign(y-B.dely)*Biy
    Biz = sign(z)*Biz
    B, Bqx, Bqy, Bqz, Bix, Biy, Biz = sqrt((Bqz+Biz)^2+(Bqy+Biy)^2+(Bqx+Bix)^2), Bqx, Bqy, Bqz, Bix, Biy, Biz
    if x == 0 
        Bqx = 0
    end
    if y == 0 
        Bqy = 0 
    end
    if z == 0 
        Bqz = 0 
    end

    return [Bqx + Bix, Bqy + Biy, Bqz + Biz]
end

function Energy(B::BfieldMag, x,y,z)
    return mz*uRb87*Bnorm(B,x,y,z)
end

function bqz(B::BfieldMag,n, m, dq, coord)
    x,y,z = coord
    radial, Radius, Zdist = getVar(B, x, y, z, n, m, dq)
    kqi = kq(B,radial, Radius, Zdist)
    return u0*B.Iq0/(2*π*((Radius+radial)^2+Zdist^2)^(1/2)) *(ellipk(kqi^2) + (Radius^2-radial^2-Zdist^2)/((Radius-radial)^2+Zdist^2)*ellipe(kqi^2))
end

function bqx(B::BfieldMag, n, m, dq, coord)
    x,y,z = coord
    radial, Radius, Zdist = getVar(B, x, y, z, n, m, dq)
    kqi = kq(B,radial, Radius, Zdist)
    return u0*B.Iq0*Zdist/(2*π*radial*((Radius+radial)^2+Zdist^2)^(1/2)) *(-ellipk(kqi^2) + (Radius^2+radial^2+Zdist^2)/((Radius-radial)^2+Zdist^2)*ellipe(kqi^2))*cos(atan(y/x))
end

function bqy(B::BfieldMag, n, m, dq, coord)
    x,y,z = coord
    radial, Radius, Zdist = getVar(B, x, y, z, n, m, dq)
    kqi = kq(B,radial, Radius, Zdist)
    return u0*B.Iq0*Zdist/(2*π*radial*((Radius+radial)^2+Zdist^2)^(1/2)) *(-ellipk(kqi^2) + (Radius^2+radial^2+Zdist^2)/((Radius-radial)^2+Zdist^2)*ellipe(kqi^2))*sin(atan(y/x))
end

function bix(B::BfieldMag,n, m, dq, coord)
    x,y,z = coord
    radial, Radius, Zdist = getVari(B, x, y, z, n, m, dq)
    kqi = kqI(B,radial, Radius, Zdist)
    return u0*B.Ii0*Zdist/(2*π*radial*((Radius+radial)^2+Zdist^2)^(1/2)) *(-ellipk(kqi^2) + (Radius^2+radial^2+Zdist^2)/((Radius-radial)^2+Zdist^2)*ellipe(kqi^2))*cos(atan((y-B.dely)/z))
end

function biy(B::BfieldMag, n, m, dq, coord)
    x,y,z = coord
    radial, Radius, Zdist = getVari(B, x, y, z, n, m, dq)
    kqi = kqI(B,radial, Radius, Zdist)
    return u0*B.Ii0*Zdist/(2*π*radial*((Radius+radial)^2+Zdist^2)^(1/2)) *(-ellipk(kqi^2) + (Radius^2+radial^2+Zdist^2)/((Radius-radial)^2+Zdist^2)*ellipe(kqi^2))*sin(atan((y-B.dely)/z))
end

function biz(B::BfieldMag,n, m, dq, coord)
    x,y,z = coord
    radial, Radius, Zdist = getVari(B, x, y, z, n, m, dq)
    kqi = kqI(B,radial, Radius, Zdist)
    return u0*B.Ii0/(2*π*((Radius+radial)^2+Zdist^2)^(1/2)) *(ellipk(kqi^2) + (Radius^2-radial^2-Zdist^2)/((Radius-radial)^2+Zdist^2)*ellipe(kqi^2))
end

end 