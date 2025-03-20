# %% imports



import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.fft import rfft,irfft,rfftfreq
from scipy.special import lambertw
from scipy.interpolate import CubicSpline as CS
from matplotlib import cm as cmap
import random
from scipy.fft import rfft,irfft,rfftfreq



        
# %% recirculation bubble
    
def calchrec(hh,xh,arec): #recirculation bubble
    dhdx = np.diff(hh)/dxh 
    tr = np.where(dhdx<np.tan(arec))[0] #where slope > 14 degrees
    hr = np.copy(hh)
    if len(tr)>0: #if there are places to add bubble
        temp = (xh[:,None]-xh[tr])
        temp[temp<=0] = np.nan
        temp = temp*np.tan(arec)+hh[tr]
        hr[np.any(temp>hh[:,None],1)] = np.nan
        xrf,hrf = xh[~np.isnan(hr)],hr[~np.isnan(hr)]
        if xrf[-1]<xh[-1]:
            xrf = np.hstack((xrf,xrf[-1]-hrf[-1]/np.tan(arec),xrf[-1]-hrf[-1]/np.tan(arec)+dxh))
            hrf = np.hstack((hrf,0,0))
        hr[np.isnan(hr)] = CS(xrf,hrf)(xh[np.isnan(hr)]) #try the cs hermit spline to speed up  this part
    return np.clip(hr,hh,None)    
 

# %% params
 

def calv_eff(usth, z_1,z_0,z_m, u_star,kap):
    return((usth/kap)*(np.log(z_1/z_0)+(z_1/z_m)*(u_star/usth)))


def calu_s(v_eff, u_f,e_t, hh,alpha,dxh):
    grad_h = (hh - np.roll(hh, 1))/dxh
    A_us = abs(e_t+2*alpha*grad_h)
    return ((v_eff-(u_f/(A_us*(2*alpha)**0.5)))*e_t-grad_h*(((2*alpha)**0.5)*u_f)/A_us)
    
    
def callsat(z_1,z_0,z_m, u_f,e_t,alpha,gamma,g,u_star,usth, hh, kap, dxh):
    #variable for the dune field, size of hh
    v_eff = calv_eff(usth, z_1,z_0,z_m, u_star,kap)
    u_s = calu_s(v_eff,u_f,e_t, hh,alpha,dxh)
    return ((2*alpha*abs(u_s)**2)/gamma*g)*(1/(((u_star/usth)**2)-1))

# %% constants


# 
gam   = 0.57721 # euler's number, ND
kap   = 0.4 # von karman's constant, ND
z0    = 1e-3 # roughness length, m
arec  = -14*np.pi/180 # recirculation bubble initiation slope
aav   = -34*np.pi/180 # static avalance slope
dav   = -36*np.pi/180 # dynamic avalance slope
d     = 255e-6 # grain size, m
rhos  = 2650 # sediment density, kg/m^3
rhof  = 1.2 # fluid density, kg/m^3
g     = 9.81 # gravity, m/s^2
z_1   = 3e-3 # refernce height
z_0   = 1e-6 # grain roughness lengh, m
z_m   = 2e-2 # height of saltation layer
alpha = 0.35 # effective resolution coeficient, from doi:10.1088/1742-5468/2006/07/P07011 fig.4 

## don't understand formula
gamma = 1 # splash porcess parmeter from DOI: 10.1103/PhysRevE.64.031305

## have not rhoguht about how it'll change in 3d, might need to adjust cte type or make it dynamic
e_t = 1 # wind direction

## don;t know where to get this value from, should eb a constant
u_f= 1 #grain settling velocity m/s


usth  = 0.082*(g*d*rhos/rhof)**0.5 # threshold friction velocity, m/s


tauth = usth**2*rhof # threshold shear stress, kg/m/m^2
lsat  = 2.2*d*rhos/rhof # saltation saturation length, m
qsat0 = 8.3*rhof/rhos*usth**3/g # saturated volumetric sediment flux per unit width at threshold, m^2/s


# grid
Tx  = 300 # non-dimensional domain size
dxh = 1e-0 # non-dimensional grid step (numerically stable ~< 1e-1)
Nxh = int(Tx/dxh) # domain size

xh = dxh*np.arange(Nxh) # non-dimensional spatial grid
x  = xh*lsat # spatial grid, m

# boundary conditions
tau00 = 0.3 # average above-threshold entry shear stress, kg/m/s^2
tau0 = np.full_like(xh,tau00)
#tau0 = np.linspace(tau00,tau00/3,len(xh)) # impose idealised linear decrease in far-field shear stress with distance to induce deposition
qi0 = (tau00/tauth-1) # influx to be at saturated flux for entry shear stress


u_star = ((tau0/rhof)**0.5)*((1+tauth)**0.5) #perturbed shear velocity 

# run time
Tt = 300 # non-dimensional simulation duration

qintfac = 5e-2 # fraction of real time that shear stress exceeds threshold (e.g., sand only moving ~5% of the year)
dhdtsat = (tau0[0]-tau0[1])/tauth/dxh

# %% dune definition
avtol = 1e-5 #theshold of change for avalanche to stop

bd=np.full_like(xh,0) #bedrock array


hh=np.full_like(xh,0)
hh  =  4 * np.exp(-((xh - 50) / 12) ** 2) #big dune


hh= np.maximum(bd,hh)
hh0=np.copy(hh) #initial topogrpahy
Nth = 10
dth = Tt/Nth
th = np.linspace(0,Tt,Nth+1)
hs = np.zeros((len(th),len(xh)))*np.nan
hs[0] = hh

# %% runs model

lsat= callsat(z_1,z_0,z_m, u_f,e_t,alpha,gamma,g,u_star,usth, calchrec(hh,xh,arec), kap,dxh)

# %% plots

fig, ax1 = plt.subplots()
ax1.plot(lsat, label="LSAT", color="b")
ax1.set_xlabel("Index")  # X-axis label
ax1.set_ylabel("LSAT", color="b")
ax1.tick_params(axis="y", labelcolor="b")
plt.legend()

# Create second y-axis
ax2 = ax1.twinx()
ax2.plot(hh, label="HH", color="r", linestyle="--")
ax2.set_ylabel("Dune height (m)", color="r")
ax2.tick_params(axis="y", labelcolor="r")

plt.legend()
plt.text(200,2.5, "u_f = "+str(u_f))
plt.show()
