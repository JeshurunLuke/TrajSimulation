import numpy as np
#==============================================================
# Fixed stepsize integrator.
# Containing:
#   ode_fixed
#   ode_bvp
#==============================================================
# function [x,y] = ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep)
#
# Solving a system of 
# ordinary differential equations using fixed
# step size. 
#
# input:
#   nstep  : number of steps
#   fRHS   : function handle. Needs to return a vector of size(y0).
#   fORD   : function handle. integrator order (step for single update). 
#   fBVP   : unused function handle. For consistency in calls by ode_test.
#   x0     : starting x.
#   y0     : starting y (this is a (nvar,1) vector).
#   x1     : end x.
#
# output:
#   x      : positions of steps (we'll need this for
#            consistency with adaptive step size integrators later)
#   y      : (nvar,nstep+1) maxtrix of resulting y's
#---------------------------------------------------------------

def ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep,**kwargs):
    try: 
        SC = kwargs['StoppingCondition']
    except: 
        SC = lambda pos: 1
    if (((fRHS.__name__ == 'keplerdirect_symp1') or (fRHS.__name__ == 'keplerdirect_symp2')) and (not (fORD.__name__ == 'euler'))):
        raise Exception('[ode_ivp]: symplectic integrators require fORD = step_euler.')

    nvar    = y0.size                      # number of ODEs
    x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
    y       = np.zeros((nvar,nstep+1))     # result array 
    y[:,0]  = y0                           # set initial condition
    dx      = x[1]-x[0]                    # step size
    it      = np.zeros(nstep+1)
    #print("[ode_ivp]: k=%5i x=%13.5e y=%13.5e %13.5e %13.5e %13.5e" % (0,x[0],y[0,0],y[1,0],y[2,0],y[3,0]))
    for k in range(1,nstep+1):
        y[:,k],it[k] = fORD(fRHS,x[k-1],y[:,k-1],dx,**kwargs)
        if SC(y[:, k]) == -1:
            return x[0:k],y[:,0:k],k

        #print("[ode_ivp]: k=%5i x=%13.5e y=%13.5e %13.5e %13.5e %13.5e R=%13.5e" % (k,x[k],y[0,k],y[1,k],y[2,k],y[3,k],np.sqrt(y[0,k]*y[0,k]+y[1,k]*y[1,k])))
    return x,y,it

#==============================================================
# function [x,y] = ode_bvp(fRHS,fORD,fBVP,x0,y0,x1g,nstep,**kwargs)
#
# Solving an open-range BVP. Assumes that fRHS does not depend
# explicitly on x.
#
# input:
#   fRHS   : function handle. Needs to return a vector of size(y0).
#   fORD   : function handle. Integrator step function.
#   fBVP   : function handle. Returns -1,0,1 depending on whether
#            range needs to be shortened, is ok (within accuracy) or extended.
#   x0     : starting x.
#   y0     : starting y (this is a (nvar,1) vector).
#   x1g    : guess for end x
#   nstep  : number of steps
#
# output:
#   x      : positions of steps (we'll need this for
#            consistency with adaptive step size integrators later)
#   y      : (nvar,nstep+1) maxtrix of resulting y's
# 
# Note: 
#   (1) Solves an open integration range BVP in the independent variable 
#       x. x0 is specified, but x1 is unknown. 
#   (2) Calls ode_ivp to solve extended IVP.
#   (3) This is a special (simplified) version of an open-range BVP. 
#       We assume that fRHS does not depend explicitly on x.
# 
#---------------------------------------------------------------

def ode_bvp(fRHS,fORD,fBVP,x0,y0,x1g,nstep,**kwargs):

    eps     = 1e-5
    nvar    = y0.size
    x1      = x1g
    x2      = x1

    # (1) root bracketing
    x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep,**kwargs)
    f1     = fBVP(y[:,nstep])
    f2     = f1
    if (f1 > 0): # increase x1
        xfac = 1.10
    else:        # decrease x1
        xfac = 0.90
    while (f1*f2 > 0.0): # bracket the root
        f1     = f2
        x1     = x2
        x2     = xfac*x2
        x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,x2,nstep,**kwargs)
        f2     = fBVP(y[:,nstep])
    if (x1 > x2):     # rearrange such that x1<x2
        x1,x2 = x2,x1 # tuple swap 
        f1,f2 = f2,f1
    # (2) root finder
    xm  = 0.5*(x1+x2)
    x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,xm,nstep,**kwargs) 
    fm  = fBVP(y[:,nstep])
    while (np.abs(x2-x1)/xm > eps):
        if (f1*fm > 0.0):
            x1 = xm
            f1 = fm
        else:
            x2 = xm
        xm     = 0.5*(x1+x2)
        x,y,it = ode_ivp(fRHS,fORD,fBVP,x0,y0,xm,nstep,**kwargs)
        fm     = fBVP(y[:,nstep])

    return x,y,it
    

