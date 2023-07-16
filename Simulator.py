import numpy as np
from arc import *                 #Import ARC (Alkali Rydberg Calculator)
import numpy as np
from joblib import Parallel, delayed

from scipy.integrate import solve_ivp
from scipy.linalg import expm, norm
from numpy import cross, eye, dot
from scipy.spatial import cKDTree
import scipy.stats as st
Consts = {'c': 2.99798458E8, 'hbar':1.054571E-34, 'kb':1.3806452E-23, 'e0':8.85418E-12}
Conversion = {'mK2J': Consts['kb']*1E-3, 'J2mK': 1/(Consts['kb']*1E-3)}


#UTILITIES

pol_v = {'sigma_p': -1/np.sqrt(2)*np.array([1, 1j, 0]), 'sigma_m': 1/np.sqrt(2)*np.array([1, -1j, 0]), 'pi': np.array([0, 0,  1])}

def rotation_matrix(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))
def get_angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(norm(v1)*norm(v2)))
def decompose_spherical(pol): 
    return np.array([np.vdot(pol_v['pi'], pol), np.vdot(pol_v['sigma_p'], pol), np.vdot(pol_v['sigma_m'], pol)])
def project_onto_plane(x, n):
    n = n/norm(n)
    return x - np.dot(x, n)*n
def project_onto_vector(x, n):
    n = n/norm(n)
    return np.dot(x, n)*n
def getSphereVec(n):
    uvec = st.norm.rvs(size=n)
    uvec = uvec/np.sqrt(np.sum(uvec**2))    
    return uvec

from dataclasses import dataclass

@dataclass
class BeamProperties: 
    Loc: np.array
    dir: np.array
    pol: np.array
    waist: float = 1E10
    zr: float = 1E10

class CesiumWrapper(Cesium): 
    def __init__(self):
        self.atom = Cesium()
        self.eV2Hz = 2.417989242E14
    def findState(States, F, mf): 
        ind = np.intersect1d(np.where(States[1] == F),  np.where(States[2] == mf))
        if len(ind) > 1:
            return "Error"
        Energy = States[0]
        return Energy[0, ind[0]], ind[0]
    def Energy(self, n, l, j, f, mF, B): 
        States = self.atom.breitRabi(n, l, j, np.array([B]))
        return   CesiumWrapper.findState(States, f, mF)[0] + self.atom.getEnergy(n, l, j)*self.eV2Hz 
    def getw0(self, stateI, stateF, B):
        return self.Energy(*stateF, B)  - self.Energy(*stateI, B) 
    def getSaturationIntensity(self, stateI, stateF):
        return self.atom.getSaturationIntensity(*stateI, *stateF)
    def getTransitionRate(self, stateI, stateF):
        return self.atom.getTransitionRate(*stateF[0:3], *stateI[0:3])
class GaussianBeam: 
    def Intensity(I0, pos, BP):
        r = norm(project_onto_plane(pos, BP.dir))
        z = norm(project_onto_vector(pos, BP.dir) - BP.Loc)
        w = lambda z: BP.waist*np.sqrt(1  + (z/BP.zr)**2)
        I = I0*(BP.waist/w(z))**2*np.exp(-2*r**2/(w(z)**2))
        return I
    def get_raleighLength(w0, wavelength):
        return np.pi*w0**2/wavelength
        
class Light:
    def __init__(self, stateI, stateF, delw, I, B, BP):
        atom = CesiumWrapper()
        self.mass = 2.2E-25
        self.w0 = atom.getw0(stateI, stateF,norm(B(np.array([0, 0, 0])))) # Assumes the magnetic field at the center for the hyperfine shift
        self.wavelength = Consts['c']/(self.w0)
        self.k = 2*np.pi/self.wavelength*BP.dir
        self.delw = delw
        self.gamma = atom.getTransitionRate(stateI, stateF)
        self.Isat = atom.getSaturationIntensity(stateI, stateF )
        Rot = lambda pos: rotation_matrix(np.cross(B(pos), BP.dir), get_angle(B(pos), BP.dir))
        rotated_pol = lambda pos:  Rot(pos) @ BP.pol
        q = stateF[-1] - stateI[-1]
        self.BP = BP
        self.s0 =  lambda pos: GaussianBeam.Intensity(I, pos, BP)/self.Isat * np.abs(rotated_pol(pos)[q])**2/norm(rotated_pol(pos))**2 
    def get_scatteringrate_abs(self,pos, v):
        return self.s0(pos)*self.gamma/2*1/(1 + self.s0(pos) + (2*(self.delw + np.dot(self.k, v))/self.gamma)**2)
    def get_scatteringrate(self, pos): 
        return self.s0(pos)*self.gamma/2*1/(1 + self.s0(pos) + (2*self.delw /self.gamma)**2)
    def get_Fabs(self, pos, vel):
        return Consts['hbar']*(self.w0 + self.delw)/Consts['c']*self.get_scatteringrate_abs(pos, vel)*self.BP.dir
    def get_Fspon(self, pos): 
        return Consts['hbar']*norm(self.k)*self.get_scatteringrate(pos)*getSphereVec(3)
    def get_Fnet(self, pos, vel): 
        return self.get_Fspon(pos) + self.get_Fabs(pos, vel)
class Tweezer: 
    def __init__(self, stateI, stateF, wavelength, trapR, trapZ, BP):
        self.atom = Cesium()
        mass = self.atom.mass
        
        self.trapR = trapR
        self.trapZ = trapZ
        BP.waist = trapR/trapZ*(1/(np.pi*np.sqrt(2)))*wavelength
        BP.zr = GaussianBeam.get_raleighLength(BP.waist, wavelength)
        self.P0 = 1/8 * trapR**2 * np.pi* mass *BP.waist**4
        self.I0 = self.P0*2/(np.pi*BP.waist**2)
        self.U0 = 1/4 * mass*BP.waist**2 * trapR**2
        print(f"Trap Depth = {self.U0*Conversion['J2mK']}")
        
        
        self.PosTree = []
        self.ForceBranch = []
        self.Damping_rate = self.atom.getTransitionRate(*stateI, *stateF)
        self.omega_0 =  Consts['c']/self.atom.getTransitionWavelength(*stateI, *stateF)
        self.omega = Consts['c']/wavelength
        self.BP = BP
    def generateAtoms(self, AtomNum, T):
        posList = self.generateAtoms_Pos(AtomNum, T)
        velList = self.sampleVelocities_BD(AtomNum, T)
        ivs = []
        for i in range(len(posList)):
            ivs.append([*posList[i], *velList[i]])
        return np.array(ivs)      
    def generateAtoms_Pos(self, AtomNum, T): #Limitation to k in one of the axis
        coordList = []
        #Pulls the coordinate for x, y and z
        sigmaR = 1/self.trapR *np.sqrt(Consts['kb']*T/self.atom.mass)
        sigmaZ = 1/self.trapZ *np.sqrt(Consts['kb']*T/self.atom.mass)
        coordZ = np.random.normal(0, sigmaZ, AtomNum)
        coordR = np.random.normal(0, sigmaR, AtomNum)
        for i in range(AtomNum):
            x, y = coordR[i]*getSphereVec(2) 
            z = coordZ[i]
            if not np.all(self.BP.dir ==  np.array([0, 0, 1])):
                Rot = rotation_matrix(np.cross(np.array([0, 0, 1]), self.BP.dir), get_angle(np.array([0, 0, 1]), self.BP.dir))
                rotated_coord = Rot @ np.array([x, y, z])
            else:
                rotated_coord = np.array([x, y, z])
            coordList.append(rotated_coord + self.BP.Loc)
        return np.array(coordList)
    def sampleVelocities_BD(self, N, T):
        scale = np.sqrt((Consts['kb'] * T)/self.atom.mass)
        # Get N numbers from U[0,1]
        nums = st.uniform.rvs(size=N)
        # Use inverse cdf of maxwell dist:
        velocities = st.maxwell.ppf(nums, scale=scale)
        velList = []
        for i in velocities: 
            print(1/2*self.atom.mass*i**2 *Conversion['J2mK'])
            velList.append(i*getSphereVec(3) ) #Velocites?
        return velList
    def alpha(self):
        return 6*np.pi*Consts['e0']*Consts['c']**3 * (self.Damping_rate/self.omega_0**2)/(self.omega_0**2 - self.omega**2 - 1j*(self.omega**3/self.omega_0**2)*self.Damping_rate)
    def alphaDet(self): #
        return (3*np.pi*Consts['c']**2*self.Damping_rate)/(2*self.omega_0**3)*(1/(self.omega_0-self.omega) + 1/(self.omega_0 - self.omega))
    def build_ForceTree(self, grid):
        x, y, z = grid
        Points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        print("Calculating Intensities")
        Int_field = np.array(Parallel(n_jobs=10)(delayed(GaussianBeam.Intensity)(self.I0, pos, self.BP) for pos in Points))
        Imesh_r = Int_field.reshape(x.shape)
        Igrad_x, Igrad_y, Igrad_z = np.gradient(Imesh_r)
        Fx_m, Fy_m, Fz_m = 1/(2*Consts['e0']*Consts['c'])*np.real(self.alpha())*np.array([Igrad_x/np.diff(x[:, 0, 0])[0], Igrad_y/np.diff(y[0, :, 0])[0], Igrad_z/np.diff(z[0, 0, :])[0]])
        self.ForceBranch = np.array([Fx_m.ravel(), Fy_m.ravel(),Fz_m.ravel()])
    def set_PosTree(self, tree):
        print("Setting Tree")
        self.PosTree = tree
    def get_Fnet(self, pos, vel): 

        distances, indices = self.PosTree.query(pos, k=1)
        Fpot = self.ForceBranch[:, indices]
        return Fpot

class System: 
    def __init__(self, atom,  magneticField):
        self.TweezerConfig = []
        self.BeamConfig = []
        self.Points = []
        self.Tree = []
        self.grid = []
        self.atom = atom
        self.magneticField = magneticField
    def clear_beams(self):
        self.TweezerConfig = []
        self.BeamConfig = []
    def set_grid(self, xbounds, ybounds, zbounds, spacing=10):
        x_ = np.linspace(*xbounds, spacing)
        y_ = np.linspace(*ybounds, spacing)
        z_ = np.linspace(*zbounds, spacing)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        self.grid = [x, y, z]
        self.Points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        self.Tree = cKDTree(self.Points)
        print(self.Tree)
    def set_tweezer(self, stateI, stateF, wavelength, trapR, trapZ, BeamParam): 
        self.TweezerConfig.append(Tweezer(stateI, stateF, wavelength, trapR, trapZ, BeamParam))
        print("Calculate Tweezer Force")
        self.TweezerConfig[-1].build_ForceTree(self.grid)
        self.TweezerConfig[-1].set_PosTree(self.Tree)
        print("Completed")

        
    def set_beams(self, stateI, stateF, delw, I, BeamParam): 
        self.BeamConfig.append(Light(stateI, stateF, delw, I, self.magneticField, BeamParam))
    def join_beams(self):
        self.beams = self.BeamConfig + self.TweezerConfig
    def StoppingConditions(self, t, y): 
        pos_vec = [y[0], y[1], y[2]]
        vel_vec = [y[3], y[4], y[5]]
        pointT = np.transpose(self.Points)
        if min(pointT[0, :])<pos_vec[0]< max(pointT[0, :]) or  min(pointT[1, :])<pos_vec[1]< max(PointT[1, :]) or  min(PointT[2, :])<pos_vec[2]< max(PointT[2, :]):
            return -1
        else:
            return 1

def set_System(System):
    def RHS(t, y, dt): 
        pos_vec = [y[0], y[1], y[2]]
        vel_vec = [y[3], y[4], y[5]]
        Fnet = np.zeros(3)
        for beam in System.beams: 
            Fnet += beam.get_Fnet(pos_vec, vel_vec)
        
        a_vec = 1/System.atom.mass * (Fnet*dt)
        return np.array([*vel_vec, *a_vec])
    return RHS
