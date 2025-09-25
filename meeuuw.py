import numpy as np
import sys as sys
import numba
import random
import time as clock
from scipy import sparse
import scipy.sparse as sps

###############################################################################
# velocity basis functions
###############################################################################

@numba.njit
def basis_functions_V(r,s):
    N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N_1=    (1.-r**2) * 0.5*s*(s-1.)
    N_2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N_3= 0.5*r*(r-1.) *    (1.-s**2)
    N_4=    (1.-r**2) *    (1.-s**2)
    N_5= 0.5*r*(r+1.) *    (1.-s**2)
    N_6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N_7=    (1.-r**2) * 0.5*s*(s+1.)
    N_8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

@numba.njit
def basis_functions_V_dr(r,s):
    dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr_1=       (-2.*r) * 0.5*s*(s-1)
    dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr_3= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr_4=       (-2.*r) *   (1.-s**2)
    dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr_6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr_7=       (-2.*r) * 0.5*s*(s+1)
    dNdr_8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,\
                     dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

@numba.njit
def basis_functions_V_ds(r,s):
    dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds_1=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds_3= 0.5*r*(r-1.) *       (-2.*s)
    dNds_4=    (1.-r**2) *       (-2.*s)
    dNds_5= 0.5*r*(r+1.) *       (-2.*s)
    dNds_6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds_7=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds_8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,\
                     dNds_6,dNds_7,dNds_8],dtype=np.float64)

###############################################################################
# pressure basis functions 
###############################################################################

@numba.njit
def basis_functions_P(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1-r)*(1+s)
    N_3=0.25*(1+r)*(1+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

###############################################################################
# jit on this function really makes a difference - keep it
###############################################################################

@numba.njit
def interpolate_vel_on_pt(xm,ym,u,v):
    ielx=int(xm/Lx*nelx)
    iely=int(ym/Ly*nely)
    #if ielx<0: exit('ielx<0')
    #if iely<0: exit('iely<0')
    #if ielx>=nelx: exit('ielx>nelx')
    #if iely>=nely: exit('iely>nely')
    iel=nelx*iely+ielx
    xmin=x_V[icon_V[0,iel]] 
    ymin=y_V[icon_V[0,iel]] 
    rm=((xm-xmin)/hx-0.5)*2
    sm=((ym-ymin)/hy-0.5)*2
    N=basis_functions_V(rm,sm)
    um=np.dot(N,u[icon_V[:,iel]])
    vm=np.dot(N,v[icon_V[:,iel]])
    return um,vm,rm,sm,iel

@numba.njit
def interpolate_T_on_pt(xm,ym,T):
    ielx=int(xm/Lx*nelx)
    iely=int(ym/Ly*nely)
    iel=nelx*iely+ielx
    xmin=x_V[icon_V[0,iel]] 
    ymin=y_V[icon_V[0,iel]] 
    rm=((xm-xmin)/hx-0.5)*2
    sm=((ym-ymin)/hy-0.5)*2
    N=basis_functions_V(rm,sm)
    Tm=np.dot(N,T[icon_V[:,iel]])
    return Tm

@numba.njit
def locate_pt(x,y):
    ielx=int(x/Lx*nelx)
    iely=int(y/Ly*nely)
    return nelx*iely+ielx

###############################################################################
# constants

cm=0.01
eps=1e-9
year=365.25*3600*24
Myear=365.25*3600*24*1e6

print("-----------------------------")
print("----------- MEEUW -----------")
print("-----------------------------")

###############################################################################
# experiment 0: Blankenbach et al, 1993
# experiment 1: van Keken et al, JGR, 1997
# experiment 2: Schmeling et al, PEPI 2008

experiment=1

match(experiment):
     case(0):
         Lx=1
         Ly=1
         eta_ref=1
         solve_T=True
         vel_scale=1
         time_scale=1
         p_scale=1
         Ttop=0
         Tbottom=1
         alphaT=1e-2   # thermal expansion coefficient
         hcond=1      # thermal conductivity
         hcapa=1    # heat capacity
         rho0=1
         Ra=1e6
         gy=-Ra/alphaT 
         TKelvin=0
     case(1):
         Lx=0.9142
         Ly=1
         gy=-1 
         eta_ref=1
         solve_T=False
         vel_scale=1
         p_scale=1
         time_scale=1
     case(2):
         eta_ref=1e21
         p_scale=1e6
         vel_scale=cm/year
         time_scale=year
     case _ :
         exit('unknown experiment')  

if int(len(sys.argv) == 4):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   nstep = int(sys.argv[3])
else:
   nelx=64
   nely=64
   nstep=500

CFLnb=0.5

RKorder=2
nparticle_per_dim=8
random_particles=True

ndim=2                     # number of dimensions
ndof_V=2                   # number of velocity dofs per node
nel=nelx*nely              # total number of elements
nn_V=(2*nelx+1)*(2*nely+1) # number of V nodes
nn_P=(nelx+1)*(nely+1)     # number of P nodes

m_V=9 # number of velocity nodes per element
m_P=4 # number of pressure nodes per element
m_T=9 # number of temperature nodes per element

r_V=[-1,0,+1,-1,0,+1,-1,0,+1]
s_V=[-1,-1,-1,0,0,0,+1,+1,+1]

ndof_V_el=m_V*ndof_V

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem_T=nn_V        # number of temperature dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

EBA=False

debug=False

###############################################################################

t01=0 ; t02=0 ; t03=0 ; t04=0 ; t05=0 ; t06=0 ; t14=0 ; t15=0
t07=0 ; t08=0 ; t09=0 ; t10=0 ; t11=0 ; t12=0 ; t13=0 ; t16=0

###############################################################################
# quadrature rule points and weights
###############################################################################

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
nqel=nqperdim**ndim

###############################################################################
# open output files & write headers
###############################################################################

vrms_file=open('vrms.ascii',"w")
vrms_file.write("#time,vrms\n")
pstats_file=open('pressure_stats.ascii',"w")
pstats_file.write("#istep,min p, max p\n")
vstats_file=open('velocity_stats.ascii',"w")
vstats_file.write("#istep,min(u),max(u),min(v),max(v)\n")
dt_file=open('dt.ascii',"w")
dt_file.write("#time dt1 dt2 dt\n")

###############################################################################

print('Lx       =',Lx)
print('Ly       =',Ly)
print('nn_V     =',nn_V)
print('nn_P     =',nn_P)
print('nel      =',nel)
print('Nfem_V   =',Nfem_V)
print('Nfem_P   =',Nfem_P)
print('Nfem     =',Nfem)
print('nqperdim =',nqperdim)
print("-----------------------------")

###############################################################################
# build velocity nodes coordinates 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2
        y_V[counter]=j*hy/2
        counter+=1
    #end for
#end for

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("build V grid: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

nnx=2*nelx+1 
nny=2*nely+1 

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,3):
            for l in range(0,3):
                icon_V[counter2,counter]=i*2+l+j*2*nnx+nnx*k
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64) # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64) # y coordinates

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter+=1
    #end for
 #end for

if debug: np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time() - start))

###############################################################################
# build pressure connectivity array 
###############################################################################
start = clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,2):
            for l in range(0,2):
                icon_P[counter2,counter]=i+l+j*(nelx+1)+(nelx+1)*k 
                counter2+=1
            #end for
        #end for
        counter+=1
    #end for
#end for

print("build icon_P: %.3f s" % (clock.time() - start))

###############################################################################
# define velocity boundary conditions
###############################################################################
start = clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool) # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

match(experiment):

     case(0): # Blankenbach et al convection, free slip all sides
         for i in range(0,nn_V):
             if x_V[i]/Lx<eps:
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if x_V[i]/Lx>(1-eps):
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if y_V[i]/Ly<eps:
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
             if y_V[i]/Ly>(1-eps):
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

     case(1): # van Keken et al Rayleigh-Taylor instability, no slip top, bottom 
         for i in range(0,nn_V):
             if x_V[i]/Lx<eps:
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if x_V[i]/Lx>(1-eps):
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
             if y_V[i]/Ly<eps:
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
             if y_V[i]/Ly>(1-eps):
                bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
                bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

     case _ :
         exit('unknown experiment')  

print("velocity b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

if solve_T:

   bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
   bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

   match(experiment):
        case(0):
            for i in range(0,nn_V):
                if y_V[i]<eps:
                   bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
                if y_V[i]>(Ly-eps):
                   bc_fix_T[i]=True ; bc_val_T[i]=Ttop
        case _:
            exit('unknown experiment')  

   print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)

if solve_T:

   match(experiment):
        case(0):
            for i in range(0,nn_V):
                T[i]=(Tbottom-Ttop)*(Ly-y_V[i])/Ly+Ttop\
                    -0.01*np.cos(np.pi*x_V[i]/Lx)*np.sin(np.pi*y_V[i]/Ly)
        case _:
            exit('unknown experiment')  

   T_mem=T.copy()

   if debug: np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

   print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

area=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
# compute jacobian matrix (inverse and determinant)
###############################################################################

jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcob=hx*hy/4

###############################################################################
# precompute basis functions values at q points
###############################################################################
start=clock.time()

N_V=np.zeros((nqel,m_V),dtype=np.float64) 
N_P=np.zeros((nqel,m_P),dtype=np.float64) 
dNdr_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNds_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdx_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdy_V=np.zeros((nqel,m_V),dtype=np.float64) 

rq=np.zeros(nqel,dtype=np.float64) 
sq=np.zeros(nqel,dtype=np.float64) 
weightq=np.zeros(nqel,dtype=np.float64) 
   
counterq=0 
for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):

        rq[counterq]=qcoords[iq]
        sq[counterq]=qcoords[jq]
        weightq[counterq]=qweights[iq]*qweights[jq]

        N_V[counterq,0:m_V]=basis_functions_V(rq[counterq],sq[counterq])
        N_P[counterq,0:m_P]=basis_functions_P(rq[counterq],sq[counterq])
        dNdr_V[counterq,0:m_V]=basis_functions_V_dr(rq[counterq],sq[counterq])
        dNds_V[counterq,0:m_V]=basis_functions_V_ds(rq[counterq],sq[counterq])
        dNdx_V[counterq,0:m_V]=jcbi[0,0]*dNdr_V[counterq,0:m_V]
        dNdy_V[counterq,0:m_V]=jcbi[1,1]*dNds_V[counterq,0:m_V]
        counterq+=1

print("compute N & grad(N) at q pts: %.3f s" % (clock.time()-start))

###############################################################################
# precompute basis functions values at V nodes
###############################################################################
start=clock.time()

N_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
N_P_n=np.zeros((m_V,m_P),dtype=np.float64) 
dNdr_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNds_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNdx_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNdy_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
   
for i in range(0,m_V):
    rq=r_V[i]
    sq=s_V[i]
    N_V_n[i,0:m_V]=basis_functions_V(rq,sq)
    N_P_n[i,0:m_P]=basis_functions_P(rq,sq)
    dNdr_V_n[i,0:m_V]=basis_functions_V_dr(rq,sq)
    dNds_V_n[i,0:m_V]=basis_functions_V_ds(rq,sq)
    dNdx_V_n[i,0:m_V]=jcbi[0,0]*dNdr_V_n[i,0:m_V]
    dNdy_V_n[i,0:m_V]=jcbi[1,1]*dNds_V_n[i,0:m_V]

print("compute N & grad(N) at V nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute array for assembly
###############################################################################
start=clock.time()

local_to_globalV=np.zeros((ndof_V_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
                 
print("compute local_to_globalV: %.3f s" % (clock.time()-start))

###############################################################################
# fill I,J arrays
###############################################################################
start=clock.time()

bignb=nel*( (m_V*ndof_V)**2 + 2*(m_V*ndof_V*m_P) )

II_V=np.zeros(bignb,dtype=np.int32)    
JJ_V=np.zeros(bignb,dtype=np.int32)    
VV_V=np.zeros(bignb,dtype=np.float64)    

counter=0
for iel in range(0,nel):
    for ikk in range(ndof_V_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndof_V_el):
            m2=local_to_globalV[jkk,iel]
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
        for jkk in range(0,m_P):
            m2 =icon_P[jkk,iel]+Nfem_V
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
            II_V[counter]=m2
            JJ_V[counter]=m1
            counter+=1

print("fill II_V,JJ_V arrays: %.3f s" % (clock.time()-start))

###############################################################################
# fill I,J arrays
###############################################################################
start=clock.time()

if solve_T:

   bignb=nel*m_T**2 

   II_T=np.zeros(bignb,dtype=np.int32)    
   JJ_T=np.zeros(bignb,dtype=np.int32)    
   VV_T=np.zeros(bignb,dtype=np.float64)    

   counter=0
   for iel in range(0,nel):
       for ikk in range(m_T):
           m1=icon_V[ikk,iel]
           for jkk in range(m_T):
               m2=icon_V[jkk,iel]
               II_T[counter]=m1
               JJ_T[counter]=m2
               counter+=1

   print("fill II_T,JJ_T arrays: %.3f s" % (clock.time()-start))

###############################################################################
# particle coordinates setup
###############################################################################
start=clock.time()

nparticle_per_element=nparticle_per_dim**2
nparticle=nel*nparticle_per_element

swarm_x=np.zeros(nparticle,dtype=np.float64)
swarm_y=np.zeros(nparticle,dtype=np.float64)
swarm_u=np.zeros(nparticle,dtype=np.float64)
swarm_v=np.zeros(nparticle,dtype=np.float64)
swarm_active=np.zeros(nparticle,dtype=bool) ; swarm_active[:]=True

if random_particles:
   counter=0
   for iel in range(0,nel):
       for im in range(0,nparticle_per_element):
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           N=basis_functions_V(r,s)
           swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
           swarm_y[counter]=np.dot(N[:],y_V[icon_V[:,iel]])
           counter+=1
       #end for
   #end for
else:
   counter=0
   for iel in range(0,nel):
       for j in range(0,nparticle_per_dim):
           for i in range(0,nparticle_per_dim):
               r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
               s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
               N=basis_functions_V(r,s)
               swarm_x[counter]=np.dot(N[:],x_V[icon_V[:,iel]])
               swarm_y[counter]=np.dot(N[:],y_V[icon_V[:,iel]])
               counter+=1
           #end for
       #end for
   #end for

print("     -> nparticle %d " % nparticle)
print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("particles setup: %.3f s" % (clock.time()-start))

###############################################################################
# particle paint
###############################################################################
start=clock.time()

swarm_paint=np.zeros(nparticle,dtype=np.int32)

for i in [0,2,4,6,8,10,12,14]:
    dx=Lx/16
    for im in range (0,nparticle):
        if swarm_x[im]>i*dx and swarm_x[im]<(i+1)*dx:
           swarm_paint[im]+=1

for i in [0,2,4,6,8,10,12,14]:
    dy=Ly/16
    for im in range (0,nparticle):
        if swarm_y[im]>i*dy and swarm_y[im]<(i+1)*dy:
           swarm_paint[im]+=1

print("particles paint: %.3f s" % (clock.time()-start))

###############################################################################
# particle layout
###############################################################################
start=clock.time()

swarm_mat=np.zeros(nparticle,dtype=np.int32)

match(experiment):
     case(0):
         swarm_mat[:]=1
     case(1):
         for im in range (0,nparticle):
             if swarm_y[im]<0.2+0.02*np.cos(swarm_x[im]*np.pi/0.9142):
                swarm_mat[im]=1
             else:
                swarm_mat[im]=2
     #case(2):
     case _ :
         exit('unknown experiment')  

print("     -> swarm_mat (m,M) %e %e " %(np.min(swarm_mat),np.max(swarm_mat)))

print("particle layout: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################
###############################################################################
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

geological_time=0.

topstart=clock.time()

for istep in range(0,nstep):
    print("-------------------------------------")
    print("istep= %d | time= %.4e " %(istep,geological_time))
    print("-------------------------------------")

    ###########################################################################
    # evaluate density and viscosity on particles
    # experiment 0: all isoviscous eta=1, rho0=1
    # experiment 1: all isoviscous, bottom rho1, top rho2
    # experiment 2: 3 materials
    ###########################################################################
    start=clock.time()

    swarm_T=np.zeros(nparticle,dtype=np.float64)
    swarm_rho=np.zeros(nparticle,dtype=np.float64)
    swarm_eta=np.zeros(nparticle,dtype=np.float64)

    match(experiment):
         case(0):
             for ip in range(0,nparticle):
                 swarm_T[ip]=interpolate_T_on_pt(swarm_x[ip],swarm_y[ip],T)
                 swarm_rho[ip]=rho0*(1-alphaT*swarm_T[ip])
                 swarm_eta[ip]=1
         case(1):
             for im in range(0,nparticle):
                 if swarm_mat[im]==1:
                    swarm_rho[im]=1000
                    swarm_eta[im]=100
                 else:
                    swarm_rho[im]=1010
                    swarm_eta[im]=100

         #case(2):
         case _ :
            exit('unknown experiment')  

    print("     -> swarm_T   (m,M) %e %e " %(np.min(swarm_T),np.max(swarm_T)))
    print("     -> swarm_rho (m,M) %e %e " %(np.min(swarm_rho),np.max(swarm_rho)))
    print("     -> swarm_eta (m,M) %e %e " %(np.min(swarm_eta),np.max(swarm_eta)))

    print("compute rho,eta on particles: %.3fs" % (clock.time()-start))

    t15+=clock.time()-start

    ###########################################################################
    # project particle properties on mesh
    ###########################################################################
    start=clock.time()

    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    nparticle_elemental=np.zeros(nel,dtype=np.float64) 

    for ip in range(0,nparticle):
        iel=locate_pt(swarm_x[ip],swarm_y[ip])
        rho_elemental[iel]+=swarm_rho[ip] # arithmetic 
        eta_elemental[iel]+=swarm_eta[ip] # arithmetic 
        nparticle_elemental[iel]+=1

    rho_elemental/=nparticle_elemental
    eta_elemental/=nparticle_elemental

    print("project particle fields on mesh: %.3fs" % (clock.time()-start))

    t16+=clock.time()-start

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start=clock.time()

    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
    N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
    rhs=np.zeros(Nfem,dtype=np.float64)     # right hand side of Ax=b

    counter=0
    for iel in range(0,nel):

        f_el=np.zeros((ndof_V_el),dtype=np.float64)
        K_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
        G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nqel):

            JxW=jcob*weightq[iq]

            xq=np.dot(N_V[iq,:],x_V[icon_V[:,iel]])
            yq=np.dot(N_V[iq,:],y_V[icon_V[:,iel]])
            Tq=np.dot(N_V[iq,:],T[icon_V[:,iel]])

            for i in range(0,m_V):
                dNdx=dNdx_V[iq,i] 
                dNdy=dNdy_V[iq,i] 
                B[0,2*i  ]=dNdx
                B[1,2*i+1]=dNdy
                B[2,2*i  ]=dNdy
                B[2,2*i+1]=dNdx

            #K_el+=B.T.dot(C.dot(B))*eta(Tq,xq,yq,eta0)*JxW
            K_el+=B.T.dot(C.dot(B))*eta_elemental[iel]*JxW

            for i in range(0,m_V):
                #f_el[ndof_V*i+1]+=N_V[iq,i]*JxW*rho(rho0,alphaT,Tq,T0)*gy
                f_el[ndof_V*i+1]+=N_V[iq,i]*JxW*rho_elemental[iel]*gy

            N_mat[0,0:m_P]=N_P[iq,0:m_P]
            N_mat[1,0:m_P]=N_P[iq,0:m_P]

            G_el-=B.T.dot(N_mat)*JxW

        # end for iq

        G_el*=eta_ref/Lx

        # impose b.c. 
        for ikk in range(0,ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,ndof_V_el):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
               K_el[ikk,:]=0
               K_el[:,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0

        # assemble matrix and right hand side
        for ikk in range(ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            for jkk in range(ndof_V_el):
                VV_V[counter]=K_el[ikk,jkk]
                counter+=1
            for jkk in range(0,m_P):
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
            rhs[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            rhs[Nfem_V+m2]+=h_el[k2]

    print("build FE matrix: %.3fs" % (clock.time()-start))

    t01+=clock.time()-start

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sparse_matrix=sparse.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    t02+=clock.time()-start

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/Lx)

    print("     -> u (m,M) %e %e " %(np.min(u)/vel_scale,np.max(u)/vel_scale))
    print("     -> v (m,M) %e %e " %(np.min(v)/vel_scale,np.max(v)/vel_scale))
    print("     -> p (m,M) %e %e " %(np.min(p)/p_scale,np.max(p)/p_scale))

    vstats_file.write("%10e %10e %10e %10e %10e\n" % (istep,np.min(u)/vel_scale,np.max(u)/vel_scale,\
                                                            np.min(u)/vel_scale,np.max(u)/vel_scale))

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    t14+=clock.time()-start

    print("split vel into u,v: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute timestep
    ###########################################################################
    start=clock.time()

    dt1=CFLnb*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))
    print('     -> dt1= %.6f' %(dt1/time_scale))
    
    if solve_T:
       dt2=CFLnb*(Lx/nelx)**2/(hcond/hcapa/rho0)
       print('     -> dt2= %.6f' %(dt2/time_scale))
    else:
       dt2=1e50
    dt=np.min([dt1,dt2])

    geological_time+=dt

    print('     -> dt = %.6f' %(dt/time_scale))
    print('     -> geological time = %e ' %(geological_time/time_scale))

    dt_file.write("%e %e %e %e\n" % (geological_time,dt1,dt2,dt)) ; dt_file.flush()

    print("compute time step: %.3f s" % (clock.time()-start))

    ###########################################################################
    # normalise pressure: simple approach to have <p> @ surface = 0
    ###########################################################################
    start=clock.time()

    #pressure_avrg=0
    #for iel in range(0,nel):
    #    for iq in range(0,nqel):
    #        pressure_avrg+=np.dot(N_P[iq,:],p[icon_P[:,iel]])*jcob*weightq[iq]
    #p-=pressure_avrg/Lx/Ly

    pressure_avrg=np.sum(p[nn_P-1-(nelx+1):nn_P-1])/(nelx+1)
    p-=pressure_avrg

    print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))

    pstats_file.write("%d %e %e\n" % (istep,np.min(p),np.max(p)))

    if debug: np.savetxt('p.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')
        
    t12+=clock.time()-start

    print("normalise pressure: %.3f s" % (clock.time()-start))

    ###########################################################################
    # project Q1 pressure onto Q2 (vel,T) mesh
    ###########################################################################
    start=clock.time()
    
    count=np.zeros(nn_V,dtype=np.int32)  
    q=np.zeros(nn_V,dtype=np.float64)

    for iel,nodes in enumerate(icon_V.T):
        for k in range(0,m_V):
            q[nodes[k]]+=np.dot(N_P_n[k,:],p[icon_P[:,iel]])
            count[nodes[k]]+=1
        #end for
    #end for
    
    q/=count

    print("     -> q (m,M) %.6e %.6e " %(np.min(q),np.max(q)))

    if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

    t03+=clock.time()-start

    print("compute nodal press: %.3f s" % (clock.time()-start))

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start=clock.time()

    if solve_T:

       Tvect=np.zeros(m_T,dtype=np.float64)   
       rhs=np.zeros(Nfem_T,dtype=np.float64)    # FE rhs 
       B=np.zeros((2,m_T),dtype=np.float64)     # gradient matrix B 
       N_mat=np.zeros((m_T,1),dtype=np.float64)   # shape functions

       counter=0
       for iel in range (0,nel):

           b_el=np.zeros(m_T,dtype=np.float64)
           A_el=np.zeros((m_T,m_T),dtype=np.float64)
           Ka=np.zeros((m_T,m_T),dtype=np.float64)   # elemental advection matrix 
           Kd=np.zeros((m_T,m_T),dtype=np.float64)   # elemental diffusion matrix 
           MM=np.zeros((m_T,m_T),dtype=np.float64)   # elemental mass matrix 
           velq=np.zeros((1,ndim),dtype=np.float64)

           Tvect[0:m_T]=T[icon_V[0:m_T,iel]]

           for iq in range(0,nqel):

               JxW=jcob*weightq[iq]

               N=N_V[iq,:]

               velq[0,0]=np.dot(N,u[icon_V[:,iel]])
               velq[0,1]=np.dot(N,v[icon_V[:,iel]])

               B[0,:]=dNdx_V[iq,:]
               B[1,:]=dNdy_V[iq,:]
   
               # compute mass matrix
               MM+=np.outer(N,N)*rho0*hcapa*weightq*jcob
   
               # compute diffusion matrix
               Kd+=B.T.dot(B)*hcond*JxW

               # compute advection matrix
               Ka+=np.outer(N,velq.dot(B))*rho0*hcapa*JxW

               if EBA:
                  xq=np.dot(N_V[iq,:],x_V[icon_V[:,iel]])
                  yq=np.dot(N_V[iq,:],y_V[icon_V[:,iel]])
                  Tq=np.dot(N_V[iq,:],T[icon_V[:,iel]])
                  exxq=np.dot(dNdx_V[iq,:],u[icon_V[:,iel]])
                  eyyq=np.dot(dNdy_V[iq,:],v[icon_V[:,iel]])
                  exyq=np.dot(dNdy_V[iq,:],u[icon_V[:,iel]])*0.5\
                      +np.dot(dNdx_V[iq,:],v[icon_V[:,iel]])*0.5
                  dpdxq=np.dot(dNdx_V[iq,:],q[icon_V[:,iel]])
                  dpdyq=np.dot(dNdy_V[iq,:],q[icon_V[:,iel]])
                  #viscous dissipation
                  b_el[:]+=N[:]*JxW*2*eta(Tq,xq,yq,eta0)*(exxq**2+eyyq**2+2*exyq**2) 
                  #adiabatic heating
                  b_el[:]+=N[:]*JxW*alphaT*Tq*(velq[0,0]*dpdxq+velq[0,1]*dpdyq)  
   
           #end for

           A_el+=MM+(Ka+Kd)*dt*0.5
           b_el+=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)

           # apply boundary conditions
           for k1 in range(0,m_V):
               m1=icon_V[k1,iel]
               if bc_fix_T[m1]:
                  Aref=A_el[k1,k1]
                  for k2 in range(0,m_V):
                      m2=icon_V[k2,iel]
                      b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                      A_el[k1,k2]=0
                      A_el[k2,k1]=0
                  #end for
                  A_el[k1,k1]=Aref
                  b_el[k1]=Aref*bc_val_T[m1]
               #end for
           #end for

           # assemble matrix K_mat and right hand side rhs
           for ikk in range(m_T):
               m1=icon_V[ikk,iel]
               for jkk in range(m_T):
                   VV_T[counter]=A_el[ikk,jkk]
                   counter+=1
               rhs[m1]+=b_el[ikk]
           #end for

       #end for iel

       print("build FE matrix : %.3f s" % (clock.time() - start))

       t04+=clock.time()-start

       ###########################################################################
       # solve system
       ###########################################################################
       start = clock.time()

       sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(Nfem_T,Nfem_T)).tocsr()

       T=sps.linalg.spsolve(sparse_matrix,rhs)

       print("     T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

       print("solve T time: %.3f s" % (clock.time() - start))

       t05+=clock.time()-start

    #end if solve_T

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start=clock.time()

    vrms=0.
    for iel in range(0,nel):
        for iq in range(0,nqel):
            JxW=jcob*weightq[iq]
            uq=np.dot(N_V[iq,:],u[icon_V[:,iel]])
            vq=np.dot(N_V[iq,:],v[icon_V[:,iel]])
            vrms+=(uq**2+vq**2)*JxW
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly)) 

    vrms_file.write("%e %e \n" % (geological_time/time_scale,vrms/vel_scale)) ; vrms_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms/vel_scale))

    print("compute vrms: %.3f s" % (clock.time()-start))

    t06+=clock.time()-start

    ###########################################################################
    # compute nodal heat flux 
    ###########################################################################
    start=clock.time()
    
    qx_n=np.zeros(nn_V,dtype=np.float64)  
    qy_n=np.zeros(nn_V,dtype=np.float64)  

    if istep%5==0 and solve_T: 
       count=np.zeros(nn_V,dtype=np.int32)  

       for iel in range(0,nel):
           for i in range(0,m_V):
               inode=icon_V[i,iel]
               qx_n[inode]-=np.dot(hcond*dNdx_V_n[i,:],T[icon_V[:,iel]])
               qy_n[inode]-=np.dot(hcond*dNdy_V_n[i,:],T[icon_V[:,iel]])
               count[inode]+=1
           #end for
       #end for
    
       qx_n/=count
       qy_n/=count

       print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
       print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))

    print("compute nodal heat flux: %.3f s" % (clock.time()-start))

    t07+=clock.time()-start

    ###########################################################################
    # compute Nusselt number at top
    ###########################################################################
    start=clock.time()

    if istep%2500==0 and solve_T: 

       qy_top=0
       qy_bot=0
       Nusselt=0
       for iel in range(0,nel):
           if y_V[icon_V[m_V-1,iel]]/Ly>1-eps: # top row of nodes 
              sq=+1
              for iq in range(0,nqperdim):
                  rq=qcoords[iq]
                  N=basis_functions_V(rq,sq)
                  q_y=np.dot(N,qy_n[icon_V[:,iel]])
                  Nusselt+=q_y*(hx/2)*qweights[iq]
                  qy_top+=q_y*(hx/2)*qweights[iq]
              #end for
           #end if
           if y_V[icon_V[0,iel]]/Ly<eps: # bottom row of nodes
              sq=-1
              for iq in range(0,nqperdim):
                  rq=qcoords[iq]
                  N=basis_functions_V(rq,sq)
                  q_y=np.dot(N,qy_n[icon_V[:,iel]])
                  qy_bot+=q_y*(hx/2)*qweights[iq]
           #end if
       #end for

       Nusselt=np.abs(Nusselt)/Lx

       print("     istep= %d ; Nusselt= %e " %(istep,Nusselt))

       print("compute Nu: %.3f s" % (clock.time()-start))

    t08+=clock.time()-start

    ###########################################################################
    # compute temperature profile
    ###########################################################################
    start=clock.time()

    if istep%2500==0 and solve_T: 

       T_profile=np.zeros(nny,dtype=np.float64)  
       y_profile=np.zeros(nny,dtype=np.float64)  

       counter=0    
       for j in range(0,nny):
           for i in range(0,nnx):
               T_profile[j]+=T[counter]/nnx
               y_profile[j]=y_V[counter]
               counter+=1
           #end for
       #end for

       np.savetxt('T_profile_'+str(istep)+'.ascii',np.array([y_profile,T_profile]).T,header='#y,T')

       print("compute T profile: %.3f s" % (clock.time() - start))

    t09+=clock.time()-start

    ###########################################################################
    # compute nodal strainrate
    ###########################################################################
    start=clock.time()

    if istep%5==0:   

       count=np.zeros(nn_V,dtype=np.int32)  
       e_n=np.zeros(nn_V,dtype=np.float64)  
       exx_n=np.zeros(nn_V,dtype=np.float64)  
       eyy_n=np.zeros(nn_V,dtype=np.float64)  
       exy_n=np.zeros(nn_V,dtype=np.float64)  

       for iel in range(0,nel):
           for i in range(0,m_V):
               inode=icon_V[i,iel]
               exx_n[inode]+=np.dot(dNdx_V_n[i,:],u[icon_V[:,iel]])
               eyy_n[inode]+=np.dot(dNdy_V_n[i,:],v[icon_V[:,iel]])
               exy_n[inode]+=0.5*np.dot(dNdx_V_n[i,:],v[icon_V[:,iel]])+\
                             0.5*np.dot(dNdy_V_n[i,:],u[icon_V[:,iel]])
               count[inode]+=1
           #end for
       #end for
 
       exx_n/=count
       eyy_n/=count
       exy_n/=count

       e_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

       print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
       print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
       print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))

       if debug: np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T)

    print("compute nodal sr: %.3f s" % (clock.time()-start))

    t11+=clock.time()-start

    ###########################################################################
    # advect particles
    ###########################################################################
    start=clock.time()

    if RKorder==1:

       for im in range(0,nparticle):
           if swarm_active[im]:
              swarm_u[im],swarm_v[im],rm,sm,iel=interpolate_vel_on_pt(swarm_x[im],swarm_y[im],u,v)
              swarm_x[im]+=swarm_u[im]*dt
              swarm_y[im]+=swarm_v[im]*dt
              if swarm_x[im]<0 or swarm_x[im]>Lx or swarm_y[im]<0 or swarm_y[im]>Ly:
                 swarm_active[im]=False
           # end if active
       # end for im

    elif RKorder==2:

       for im in range(0,nparticle):
           if swarm_active[im]:
              xA=swarm_x[im]
              yA=swarm_y[im]
              uA,vA,rm,sm,iel=interpolate_vel_on_pt(xA,yA,u,v)
              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Ly:
                 swarm_active[im]=False
              else:
                 uB,vB,rm,sm,iel=interpolate_vel_on_pt(xB,yB,u,v)
                 swarm_x[im]=xA+uB*dt
                 swarm_y[im]=yA+vB*dt
                 swarm_u[im]=uB
                 swarm_v[im]=vB
              # end if active
           # end if active
       # end for im

    elif RKorder==4:

       for im in range(0,nparticle):
           if swarm_active[im]:
              xA=swarm_x[im]
              yA=swarm_y[im]
              uA,vA,rm,sm,iel=interpolate_vel_on_pt(xA,yA,u,v)
              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Ly:
                 swarm_active[im]=False
              else:
                 uB,vB,rm,sm,iel=interpolate_vel_on_pt(xB,yB,u,v)
                 xC=xA+uB*dt/2.
                 yC=yA+vB*dt/2.
                 if xC<0 or xC>Lx or yC<0 or yC>Ly:
                    swarm_active[im]=False
                 else:
                    uC,vC,rm,sm,iel=interpolate_vel_on_pt(xC,yC,u,v)
                    xD=xA+uC*dt
                    yD=yA+vC*dt
                    if xD<0 or xD>Lx or yD<0 or yD>Ly:
                       swarm_active[im]=False
                    else:
                       uD,vD,rm,sm,iel=interpolate_vel_on_pt(xD,yD,u,v)
                       swarm_u[im]=(uA+2*uB+2*uC+uD)/6
                       swarm_v[im]=(vA+2*vB+2*vC+vD)/6
                       swarm_x[im]=xA+swarm_u[im]*dt
                       swarm_y[im]=yA+swarm_v[im]*dt
                    # end if active
                 # end if active
              # end if active
           # end if active
       # end for im

    else:
       exit('RKorder not available')

    for im in range(0,nparticle):
        if not swarm_active[im]:
           swarm_x[im]=0
           swarm_y[im]=0

    print('     -> nb inactive particles:',nparticle-np.sum(swarm_active))

    print("     -> swarm_x (m,M) %e %e " %(np.min(swarm_x),np.max(swarm_x)))
    print("     -> swarm_y (m,M) %e %e " %(np.min(swarm_y),np.max(swarm_y)))

    t13+=clock.time()-start

    print("advect particles: %.3f s" % (clock.time()-start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if istep%5==0: 
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f %10f %10f \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(u[i]/vel_scale,v[i]/vel_scale,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (q[i]))
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' Name='Temperature' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%e \n" %(T[i]-TKelvin))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       #for i in range(0,nn_V):
       #    vtufile.write("%e \n" %(rho(rho0,alphaT,T[i],T0)))
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='Shear heating (2*eta*e^2)' Format='ascii'> \n")
       #for i in range(0,nn_V):
       #    vtufile.write("%e \n" % (2*eta(T[i],x_V[i],y_V[i],eta0)*e_n[i]**2))
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='adiab heating (linearised)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%.15f \n" % (alphaT*T[i]*rho0*v[i]*gy))
       #vtufile.write("</DataArray>\n")
       #
       #vtufile.write("<DataArray type='Float32' Name='adiab heating (true)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%.15f \n" % (alphaT*T[i]*(u[i]*dpdx_n[i]+v[i]*dpdy_n[i]))) 
       #vtufile.write("</DataArray>\n")
       #
       #vtufile.write("<DataArray type='Float32' Name='adiab heating (diff)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%.15f \n" % (alphaT*T[i]*(u[i]*dpdx_n[i]+v[i]*dpdy_n[i])-\
       #                                alphaT*T[i]*rho0*v[i]*gy))
       #vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Heat flux' Format='ascii'> \n")
          for i in range(0,nn_V):
              vtufile.write("%e %e %e \n" %(qx_n[i],qy_n[i],0.))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (eta_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (rho_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='nb particles' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" % (nparticle_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[2,iel],icon_V[8,iel],\
                                                          icon_V[6,iel],icon_V[1,iel],icon_V[5,iel],\
                                                          icon_V[7,iel],icon_V[3,iel],icon_V[4,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m_V))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %28)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export solution to vtu file: %.3f s" % (clock.time()-start))

       t10+=clock.time()-start

    ########################################################################
    # export particles to vtu file
    ########################################################################
    start=clock.time()

    if istep%5==0 or istep==nstep-1: 

       filename = 'particles_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nparticle):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for im in range(0,nparticle):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[im]/vel_scale,swarm_v[im]/vel_scale,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='mat' Format='ascii'> \n")
       for im in range(0,nparticle):
           vtufile.write("%d \n" % swarm_mat[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for im in range(0,nparticle):
           vtufile.write("%e \n" % swarm_rho[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for im in range(0,nparticle):
           vtufile.write("%e \n" % swarm_eta[im])
       vtufile.write("</DataArray>\n")
       #--
       if solve_T:
          vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
          for ip in range(0,nparticle):
              vtufile.write("%e \n" % swarm_T[ip])
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='paint' Format='ascii'> \n")
       for ip in range(0,nparticle):
           vtufile.write("%d \n" % swarm_paint[ip])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nparticle):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nparticle):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nparticle):
           vtufile.write("%d \n" % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export particles to vtu file: %.3f s" % (clock.time() - start))

    ###########################################################################

    u_mem=u.copy()
    v_mem=v.copy()
    T_mem=T.copy()

    ###########################################################################

    if istep%25==0 or istep==nstep-1:

       duration=clock.time()-topstart

       print("-----------------------------------------------")
       print("build FE matrix V: %.3f s       | %.2f percent" % (t01,(t01/duration*100))) 
       print("solve system V: %.3f s          | %.2f percent" % (t02,(t02/duration*100))) 
       print("build matrix T: %.3f s          | %.2f percent" % (t04,(t04/duration*100))) 
       print("solve system T: %.3f s          | %.2f percent" % (t05,(t05/duration*100))) 
       print("compute vrms: %.3f s            | %.2f percent" % (t06,(t06/duration*100))) 
       print("compute nodal p: %.3f s         | %.2f percent" % (t03,(t03/duration*100))) 
       print("compute nodal sr: %.3f s        | %.2f percent" % (t11,(t11/duration*100))) 
       print("compute nodal heat flux: %.3f s | %.2f percent" % (t07,(t07/duration*100))) 
       print("compute T profile: %.3f s       | %.2f percent" % (t09,(t09/duration*100))) 
       print("export to vtu: %.3f s           | %.2f percent" % (t10,(t10/duration*100))) 
       print("normalise pressure: %.3f s      | %.2f percent" % (t12,(t12/duration*100))) 
       print("advect particles: %.3f s        | %.2f percent" % (t13,(t13/duration*100))) 
       print("split solution: %.3f s          | %.2f percent" % (t14,(t14/duration*100))) 
       print("compute swarm rho,eta: %.3f s   | %.2f percent" % (t15,(t15/duration*100))) 
       print("compute eltal rho,eta: %.3f s   | %.2f percent" % (t16,(t16/duration*100))) 
       print("-----------------------------------------------")

#end for istep

###############################################################################
# close files
###############################################################################
       
vstats_file.close()
pstats_file.close()
vrms_file.close()
dt_file.close()

###############################################################################

print("-----------------------------")
print("total compute time: %.3f s" % (duration))
print(t01+t02+t03+t04+t05+t06+t07+t08+t09+t10+t11+t12+t14+t15+t16,duration)
print("-----------------------------")
    
###############################################################################
