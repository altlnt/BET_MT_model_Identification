#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:59:50 2021

@author: alex
"""


import numpy as np
import transforms3d as tf3d
import scipy
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import scipy.optimize
import pandas as pd

# mass=369 #batterie
# mass+=1640-114 #corps-carton
# mass/=1e3
# Area=np.pi*(11.0e-02)**2d
# r0=11e-02
# rho0=1.204
# kv_motor=800.0
# pwmmin=1075.0
# pwmmax=1950.0
# U_batt=16.8

# b10=14.44

# %%   ####### IMPORT DATA 



is_sim_log=False

log_path="./logs/avion/vol123/log_real_processed.csv"
# # log_path="/home/l3x/Documents/Avion-Simulation-Identification/Logs/log_sim/2021_10_31_20h02m29s/log.txt"
save_dir=os.path.join(os.getcwd(),"res_avion_scipy_nosympy","FINAL_NEG_NOMEAN")

    
raw_data=pd.read_csv(log_path)


if not (is_sim_log):
    prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
else :
    prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) ) ])
prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])



prep_data=prep_data.reset_index()


for i in range(3):
    prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
    
    
prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
prep_data['t']-=prep_data['t'][0]
prep_data=prep_data.drop(index=[0,len(prep_data)-1])
prep_data=prep_data.reset_index()

data_prepared=prep_data[:len(prep_data)]



def scale_to_01(df):

    return (df-df.min())/(df.max()-df.min())

if is_sim_log:
    
    joystick_input=np.array([data_prepared['joystick[%i]'%(i)] for i in range(4)]).T
    
    print(joystick_input.shape)
    for i in range(len(joystick_input)):
        joystick_input[i][np.where(abs(joystick_input[i])<40)]*=0
        
        if joystick_input[i][3]/250<-0.70:
            joystick_input[i][3]=0
        else:
            joystick_input[i] = joystick_input[i]/250 * 200

    joystick_input[:,:-1] *=1.0/250 #"scale from -250,250 to -1,1"
    # joystick_input[:,:-1] += 1530  # " add mean so that it's a fake pwm value, and be processed as real logs"

        
    delta_cons= 0.5*np.array([joystick_input[:,0], -joystick_input[:,0], \
                                            (joystick_input[:,1] - joystick_input[:,2]) \
                                            , (joystick_input[:,1] + joystick_input[:,2]) ]).T
        
    delta_cons = delta_cons*500 +1530
    for i in range(1,5):
        data_prepared.insert(data_prepared.shape[1],'PWM_motor[%i]'%(i),delta_cons[:,i-1])
        
    data_prepared.insert(data_prepared.shape[1],'omega_c[5]',joystick_input[:,-1])

else:
    # data_prepared.insert(data_prepared.shape[1],'omega_c[5]',(data_prepared['PWM_motor[5]']-1000)*925.0/1000)
    data_prepared.insert(data_prepared.shape[1],'omega_c[5]',data_prepared['PWM_motor[5]']*9.57296629e-01-1.00188650e+03)
    
    radsec_reg = data_prepared['PWM_motor[5]']*9.57296629e-01-1.00188650e+03
    Tint =np.clip(5.74144050e-05*radsec_reg**2 -6.49277834e-03*radsec_reg -1.15899198e+00,0,1e10)
    # Tint =np.clip(1.5e-04*radsec_reg**2 ,0,1e10)

    data_prepared.insert(data_prepared.shape[1],'thrust_intensity',Tint)



"splitting the dataset into nsecs sec minibatches"

# %% Physical params

   

Aire_1,Aire_2,Aire_3,Aire_4,Aire_0 =    0.62*0.262* 1.292 * 0.5,\
                                    0.62*0.262* 1.292 * 0.5, \
                                    0.34*0.1* 1.292 * 0.5,\
                                    0.34*0.1* 1.292 * 0.5, \
                                    1.08*0.31* 1.292 * 0.5
                                    
Aire_list = [Aire_0,Aire_1,Aire_2,Aire_3,Aire_4]

cp_1,cp_2,cp_3,cp_4,cp_0 = np.array([-0.013,0.475,-0.040],       dtype=float).flatten(), \
                        np.array([-0.013,-0.475,-0.040],      dtype=float).flatten(), \
                        np.array([-1.006,0.17,-0.134],    dtype=float).flatten(),\
                        np.array([-1.006,-0.17,-0.134],   dtype=float).flatten(),\
                        np.array([0.021,0,-0.064],          dtype=float).flatten()
cp_list=[cp_0,cp_1,cp_2,cp_3,cp_4]

#0 : aile centrale
#1 : aile droite
#2 : aile gauche
#3 : vtail droit 
#4 : vtail gauche

theta=45.0/180.0/np.pi

Rvd=np.array([[1.0,0.0,0.0],
                [0.0,np.cos(theta),np.sin(theta)],
                [0.0,-np.sin(theta),np.cos(theta)]])

Rvg=np.array([[1.0,0.0,0.0],
                [0.0,np.cos(theta),-np.sin(theta)],
                [0.0,np.sin(theta),np.cos(theta)]])


forwards=[np.array([1.0,0,0])]*3
forwards.append(Rvd@np.array([1.0,0,0]))
forwards.append(Rvg@np.array([1.0,0,0]))

upwards=[np.array([0.0,0,1.0])]*3
upwards.append(Rvd@np.array([0.0,0,1.0]))
upwards.append(Rvg@np.array([0.0,0,1.0]))

crosswards=[np.cross(j,i) for i,j in zip(forwards,upwards)]

# Area=np.pi*(11.0e-02)**2
# r0=11e-02
# rho0=1.204
# kv_motor=800.0
# pwmmin=1075.0
# pwmmax=1950.0
# U_batt=16.8

# vwi0=0.0
# vwj0=0.0
# vwk0=0.0

alpha_0=-0.07
alpha_s = 0.3391428111
delta_s = 15.0*np.pi/180
cd0sa_0 = 0.9
cd0fp_0 = 0.010
cd1sa_0 = 2
cl1sa_0 = 5 
cd1fp_0 = 2.5 
coeff_drag_shift_0= 0.5 
coeff_lift_shift_0= 0.05 
coeff_lift_gain_0= 2.5
C_t0 = 1.1e-4 if not(is_sim_log) else 2e-4/2.0
C_q = 1e-8
C_h = 1e-4
r0=19.05
Area=r0**2*np.pi
rho0=1.292
# %% Preprocess 


df=data_prepared.copy()


df.insert(data_prepared.shape[1],
          'R',
          [tf3d.quaternions.quat2mat([i,j,k,l]) for i,j,k,l in zip(df['q[0]'],df['q[1]'],df['q[2]'],df['q[3]'])])

R_array=np.array([i for i in df["R"]])

def skew_to_x(S):
    SS=(S-S.T)/2
    return np.array([SS[1,0],SS[2,0],S[2,1]])

def skew(x):
    return np.array([[0,-x[2],x[1]],
                      [x[2],0,-x[0]],
                      [-x[1],x[0],0]])

omegas=np.zeros((R_array.shape[0],3))
omegas[1:]=[skew_to_x(j@(i.T)-np.eye(3)) for i,j in zip(R_array[:-1],R_array[1:])]
omegas[:,0]=omegas[:,0]*1.0/df['dt']
omegas[:,1]=omegas[:,1]*1.0/df['dt']
omegas[:,2]=omegas[:,2]*1.0/df['dt']

def filtering(X,k=0.05):
    Xnew=[X[0]]
    for i,x in enumerate(X[1:]):
        xold=Xnew[-1]
        xnew=xold+k*(x-xold)
        Xnew.append(xnew)
    return np.array(Xnew)

omegas_new=filtering(omegas)

v_ned_array=np.array([df['speed[%i]'%(i)] for i in range(3)]).T

v_body_array=np.array([(i.T@(j.T)).T for i,j in zip(R_array,v_ned_array)])

gamma_array=np.array([(i.T@(np.array([0,0,9.81]).T)).T for i in R_array])

for i in range(3):
    try:
        df.insert(df.shape[1],
                  'speed_body[%i]'%(i),
                  v_body_array[:,i])
    except:
        pass
    try:

        df.insert(df.shape[1],
                  'gamma[%i]'%(i),
                  gamma_array[:,i])
    except:
        pass
    try:

        df.insert(df.shape[1],
                  'omega[%i]'%(i),
                  omegas_new[:,i])
    except:
        pass

dragdirs=np.zeros((v_body_array.shape[0],3,5))
liftdirs=np.zeros((v_body_array.shape[0],3,5))
slipdirs=np.zeros((v_body_array.shape[0],3,5))

alphas=np.zeros((v_body_array.shape[0],1,5))
sideslips=np.zeros((v_body_array.shape[0],1,5))

for k,v_body in enumerate(v_body_array):
    
    
    v_in_ldp=np.cross(crosswards,np.cross((v_body-np.cross(cp_list,omegas_new[k])),crosswards))
    
    # v_in_ldp=v_body-np.cross(cp_list,omegas_new[k])
    # v_in_ldp=np.dot(v_in_ldp,upwards)*upwards+np.dot(v_in_ldp,forwards)*forwards
    # print(v_in)
    dd=-v_in_ldp
    dd=dd.T@np.diag(1.0/(np.linalg.norm(dd,axis=1)+1e-8))

    ld=np.cross(crosswards,v_in_ldp)
    ld=ld.T@np.diag(1.0/(np.linalg.norm(ld,axis=1)+1e-8))
              
    sd=-(v_body-np.cross(cp_list,omegas_new[k])-v_in_ldp)
    sd=sd.T@np.diag(1.0/(np.linalg.norm(sd,axis=1)+1e-8))
    # input("e")
    dragdirs[k,:,:]=R_array[k]@(np.diag(Aire_list)@np.diag(np.linalg.norm(v_in_ldp,axis=1))**2@(dd.T)).T
    liftdirs[k,:,:]=R_array[k]@(np.diag(Aire_list)@np.diag(np.linalg.norm(v_in_ldp,axis=1))**2@(ld.T)).T
    slipdirs[k,:,:]=R_array[k]@(np.diag(Aire_list)@np.diag(np.linalg.norm(v_in_ldp,axis=1))**2@(sd.T)).T
    
    alphas_d=np.diag(v_in_ldp@(np.array(forwards).T))/(np.linalg.norm(v_in_ldp,axis=1)+1e-8)
    alphas_d=np.arccos(alphas_d)
    alphas_d=np.sign(np.diag(v_in_ldp@np.array(upwards).T))*alphas_d
    
    x=np.linalg.norm(v_in_ldp,axis=1)
    y=np.linalg.norm(v_body-np.cross(cp_list,omegas_new[k])-v_in_ldp,axis=1)
    sideslips_d=np.arctan2(y,x)
        
    alphas[k,:,:]=alphas_d
    sideslips[k,:,:]=sideslips_d

    
alphas=alphas-np.mean(alphas,axis=0)
df.insert(df.shape[1],
          'liftdirs',
          [i for i in liftdirs])
        
df.insert(df.shape[1],
          'dragdirs',
          [i for i in dragdirs])     

df.insert(df.shape[1],
          'slipdirs',
          [i for i in slipdirs])  
        
df.insert(df.shape[1],
          'alphas',
          [i for i in alphas])    

df.insert(df.shape[1],
          'sideslips',
          [i for i in sideslips])    

df.insert(df.shape[1],
          'thrust_dir_ned',
          [i[:,0]*j**2 for i,j in zip(df['R'],df['omega_c[5]'])])

df.insert(data_prepared.shape[1],
                      'Thrust_Reg',
                      [i@np.array([0,0,j]) for i,j in zip(df['R'],df['thrust_intensity'])])


delt=np.array([df['PWM_motor[%i]'%(i)] for i in range(1,5)]).T
delt=np.concatenate((np.zeros((len(df),1)),delt),axis=1).reshape(-1,1,5)
delt=(delt-1530)/500*30.0/180.0*np.pi 
delt[:,:,0]*=0
delt[:,:,2]*=-1.0
delt[:,:,4]*=-1.0
delt=delt-np.mean(delt,axis=0)

df.insert(df.shape[1],
          'deltas',
          [i for i in delt])


# df=df[df['thrust_intensity']<1.0]
# print(df['alphas'].shape)
# print(alphas.shape)
# input("i")
# df.to_csv('./id_avion_nosympy_data.csv')
# %% plot

# drag_dirs_=  np.array([i for i in df["dragdirs"]])
# lift_dirs_=  np.array([i for i in df["liftdirs"]])


# f,axes=plt.subplots(3,1)
# cols=["darkred","darkgreen","darkblue"]

# for i in range(3):
#     axes[0].plot(df['t'],v_ned_array.reshape(-1,3)[:,i]
#              ,label=r"$v_{NED},%i$"%(i),c="rgb"[i])
    
#     axes[1].plot(df['t'],drag_dirs_[:,:,0][:,i]
#              ,label=r"$d_{ned},%i$"%(i),c="rgb"[i])
    
#     axes[2].plot(df['t'],lift_dirs_[:,:,0][:,i]
#              ,label=r"$l_{ned},%i$"%(i),c="rgb"[i])

# for j,i in enumerate(axes):
#     i.grid(),i.legend(loc=4),i.set_xlabel("t")
#     i.set_ylabel("N") if j>0 else i.set_ylabel("m/s")


# plt.figure()
# plt.plot(np.array([i for i in df['thrust_dir_ned']]),label="ned"),plt.grid(),plt.legend()


# plt.figure()
# plt.plot(v_body_array.reshape(-1,3),label="body"),plt.grid(),plt.legend()


plt.figure()
alphas=np.array([i for i in df['alphas']])
ind=(df['t']>50)*(df['t']<220)
for i in range(5):
    plt.plot(df["t"][ind],180.0/np.pi*alphas[:,0,i][ind],label=r"$\alpha_%i$"%(i))
plt.grid(),plt.legend(loc=4),plt.xlabel('t'),plt.ylabel('angle (°)')
# plt.figure()
# plt.plot(delt.reshape(-1,5),label="deltas"),plt.grid(),plt.legend()

# plt.figure()
# plt.plot(dragdirs[:,:,0],label="dragdirs")
# plt.plot(-(v_ned_array**2).reshape(-1,3),label="ned"),plt.grid(),plt.legend()
# plt.grid(),plt.legend()

# plt.figure()

# plt.plot(liftdirs[:,:,0],label="liftdirs")
# plt.grid(),plt.legend()

# df=df[df['thrust_intensity']<0.01]

# %% Regressions 



a_mw=np.mean(np.array([i for i in df['alphas']]),axis=2).flatten()
print("mean alpha :",np.mean(a_mw))
a_mw=a_mw-np.mean(a_mw)
# a_mw*=-1    

dmean=np.mean([i for i in df['deltas']],axis=2).flatten()

drag_dirs_=  np.array([i for i in df["dragdirs"]])
lift_dirs_=  np.array([i for i in df["liftdirs"]])

drag_dirs_body_= np.array([i.T@j for i,j in zip(R_array,np.sum(drag_dirs_,axis=2))])
lift_dirs_body_= np.array([i.T@j for i,j in zip(R_array,np.sum(lift_dirs_,axis=2))])

N2=drag_dirs_body_@np.array([1,0,0])

aned=np.array([df['acc_ned_grad[%i]'%(i)] for i in range(3)]).T
ab=np.array([i.T@j for i,j in zip(R_array,aned)])

Tint=df['thrust_intensity']
Y=df['acc_body_grad[0]']*8.8-df['gamma[0]']*8.8-0.57*Tint
# Y=df['acc_body_grad[0]']*8.8-df['gamma[0]']*8.8-0.56*Tint

Y/=N2

N3=lift_dirs_body_@np.array([0,0,1])

Z=df['acc_body_grad[2]']*8.8-df['gamma[2]']*8.8
Z/=N3

# # filter
# import scipy.signal
# kt=0.1
# b, a = scipy.signal.butter(1, 1./(2*np.pi)*kt,analog= False,fs=1./2.0/df['dt'].mean())
# zi = scipy.signal.lfilter_zi(b, a)
# a_mw, _ = scipy.signal.lfilter(b, a, a_mw,zi=zi*a_mw[0])
# zi = scipy.signal.lfilter_zi(b, a)
# Z, _ =  scipy.signal.lfilter(b, a, Z,zi=zi*Z[0])
# zi = scipy.signal.lfilter_zi(b, a)
# Y, _ = scipy.signal.lfilter(b, a, Y,zi=zi*Y[0])


plt.figure()
plt.plot(df['t'][ind],a_mw[ind])


yfa=np.polyfit(a_mw,Y,2)
yfa_red=yfa[0]*np.sort(a_mw)**2+yfa[1]*np.sort(a_mw)+yfa[2]

cd1sareg=yfa[0]
a0reg=-yfa[1]/2.0/cd1sareg
cdsa0reg=yfa[2]-cd1sareg*a0reg**2

# # a0reg=-a0reg
print("yfa_red",yfa)
print("cd1sareg",cd1sareg)
print("a0reg",a0reg)
print("cdsa0reg",cdsa0reg)

cd0real,cl1sareal=0.02,(0.9-0.4)/((5-0)*np.pi/180.0)
print("cd0sa sd7037",cd0real)
print("cl1sa sd7037",cl1sareal)
yfd=np.polyfit(dmean,Y,1)
yfd_red=dmean*yfd[0]+yfd[1]

def func(x,cl1sa,chi):
    return cl1sa*(x[:,0])+chi*x[:,1]

xfunc= np.c_[a_mw,dmean]
zfa, pcov = scipy.optimize.curve_fit(func,xfunc, Z)
xfunc[:,1]*=0
zfa_red=func(xfunc,*tuple(zfa))



xfunc= np.c_[a_mw,dmean]
xfunc[:,0]*=0
xfunc[:,0]+=np.mean(a_mw)
zfd_red=func(xfunc,*tuple(zfa))

def regalter(x,a0=a0reg):
    acoeff=(np.mean(Z)-0)/(np.mean(a_mw)-a0)
    return [(x-a0)*acoeff,acoeff]

def regnormale(x,a0=a0reg,cl1sa=5.75):
    return 0.5*cl1sa*np.sin(2.0*(x-a0))

coeffs,_=scipy.optimize.curve_fit(regnormale,a_mw,Z)
print("Curve fit Cl %f a0 %f"%(tuple(coeffs)))
chireg=np.polyfit(dmean,Z-regalter(a_mw)[0],1)
print("Cl1sa id interp ",regalter(a_mw)[1])
print("Chireg",chireg)
# alpha=0.01

# # %%  exploratory

# # # sans alpha 

from matplotlib import cm

alpha=0.005
f,axes=plt.subplots(1,1)

# viridis = cm.get_cmap('viridis')
scatplot=axes.scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Y,
                # alpha=alpha,
                c=Tint)
cbar=f.colorbar(cm.ScalarMappable(), ax=axes)
cbar.set_label(r'$\frac{T}{Tmax}$', rotation=0, fontsize=15)


axes.set_xlabel(r'$\alpha -\bar{\alpha}$ (°)'),axes.set_ylabel('Cd'),axes.grid()

f,axes=plt.subplots(1,1)

axes.scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Z,
                # alpha=alpha,
                c=Tint)
cbar=f.colorbar(cm.ScalarMappable(), ax=axes)
cbar.set_label(r'$\frac{T}{Tmax}$', rotation=0, fontsize=15)

axes.set_xlabel(r'$\alpha -\bar{\alpha}$ (°)'),axes.set_ylabel('Cl'),axes.grid()

# avec alpha 


f,axes=plt.subplots(1,1)

# viridis = cm.get_cmap('viridis')
scatplot=axes.scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Y,
                alpha=alpha,
                c=Tint)
cbar=f.colorbar(cm.ScalarMappable(), ax=axes)
cbar.set_label(r'$\frac{T}{Tmax}$', rotation=0, fontsize=15)


axes.set_xlabel(r'$\alpha-\bar{\alpha}$ (°)'),axes.set_ylabel('Cd'),axes.grid()

f,axes=plt.subplots(1,1)

axes.scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Z,
                alpha=alpha,
                c=Tint)
cbar=f.colorbar(cm.ScalarMappable(), ax=axes)
cbar.set_label(r'$\frac{T}{Tmax}$', rotation=0, fontsize=15)

axes.set_xlabel(r'$\alpha-\bar{\alpha}$ (°)'),axes.set_ylabel('Cl'),axes.grid()


#cl avec reg faite main


x1,y1,x2,y2=-0.15,-0.5,0.14,1.0

def reghand_2(x,x3,y3):
    return (y2-y1)/(x2-x1)*(x-x3)+y3

f,axes=plt.subplots(1,1)
print("Cl1sa : ",(y2-y1)/(x2-x1))
print("y0 :" ,y1-x1*(y2-y1)/(x2-x1))
axes.scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Z,
                # alpha=alpha,
                c=Tint)

axes.plot(180.0/np.pi*(a_mw-np.mean(a_mw)),
          reghand_2(a_mw-np.mean(a_mw),x1,y1),
          label=r"y = %.2f x + %.2f "%((y2-y1)/(x2-x1),y1-x1*(y2-y1)/(x2-x1)),
          color="r",linewidth=3)


x3,y3=x1,y1+0.5

axes.plot(180.0/np.pi*(a_mw-np.mean(a_mw)),
          reghand_2(a_mw-np.mean(a_mw),x3,y3),color="b",linestyle=":")

x3,y3=x1,y1-0.5

axes.plot(180.0/np.pi*(a_mw-np.mean(a_mw)),
          reghand_2(a_mw-np.mean(a_mw),x3,y3),color="b",linestyle=":")


axes.scatter(180.0/np.pi*x1,y1,color="black",marker="s",s=70.0)
axes.scatter(180.0/np.pi*x2,y2,color="black",marker="s",s=70.0)
cbar=f.colorbar(cm.ScalarMappable(), ax=axes)
cbar.set_label(r'$\frac{T}{Tmax}$', rotation=0, fontsize=15)

axes.set_xlabel(r'$\alpha-\bar{\alpha}$ (°)'),axes.set_ylabel('Cl'),axes.grid(),axes.legend()

# cl compraison avec / sans alpha

from matplotlib import cm

f,axes=plt.subplots(1,2)
alpha=0.005
axes[0].scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Z,
                c=Tint)
axes[0].set_xlabel(r'$\alpha-\bar{\alpha}$ (°)'),axes[0].set_ylabel('Cl'),axes[0].grid()

axes[1].scatter(180.0/np.pi*(a_mw-np.mean(a_mw)),
                Z,
                alpha=alpha,
                c=Tint)
axes[1].set_xlabel(r'$\alpha-\bar{\alpha}$ (°)'),axes[1].grid()

cbar=f.colorbar(cm.ScalarMappable(), ax=axes)
cbar.set_label(r'$\frac{T}{Tmax}$', rotation=0, fontsize=15)

# %% plot regs

# Cd regs
alpha=0.005
f,axes=plt.subplots(1,1)
axes.scatter(180/np.pi*a_mw,Y,alpha=alpha,c='black')
axes.plot(180/np.pi*np.sort(a_mw),yfa_red,label="y = %.2f $x^2$ + %.2f $x$ + %.2f"%(tuple(yfa)),color="r")
axes.scatter(180/np.pi*np.mean(a_mw),np.mean(Y),label=r"$(\bar{\alpha},\bar{C_D})$",c="green",marker="s",s=50.0)
axes.legend(),axes.set_xlabel(r'$\alpha$ (°)'),axes.set_ylabel('Cd'),axes.grid()

# # Cl regs
alpha=0.005

f,axes=plt.subplots(2,1)

axes[0].scatter(180/np.pi*a_mw,
                Z,
                alpha=alpha,
                c='black')

axes[0].plot(180/np.pi*a_mw,
              regalter(a_mw)[0],
              label=r'$y = \frac{(\bar{C_L} - 0)}{\bar{\alpha} -\alpha_{0}} \cdot (x-{\alpha_{0}} ) = %.2f (x-%.2f}$ )'%(np.round(regalter(a_mw)[1],3),np.round(cdsa0reg,3))
              ,color="r")

c1,c2=np.polyfit(a_mw,Z,1)

print("Naive reg cl1sa %f a0 %f"%(c1,-c2/c1))

axes[0].plot(180/np.pi*a_mw,
          c1*a_mw+c2,label=r'Régression naïve',
          color="purple")

axes[0].scatter(180/np.pi*np.mean(a_mw),
                np.mean(Z),c='green'
                ,marker="s",s=80.0,label=r"$(\bar{\alpha},\bar{C_L})$")

axes[0].scatter(180/np.pi*a0reg,
                0,c='orange'
                ,marker="v",s=80.0,label=r"$(\alpha_{0},0)$")


axes[0].legend(),axes[0].set_xlabel(r'$\alpha$ (°)'),axes[0].set_ylabel('Cl'),axes[0].grid()

zfd_red=chireg[0]*dmean+chireg[1]

axes[1].scatter(180/np.pi*np.mean(dmean),np.mean(Z-regalter(a_mw)[0]),
                c='black',label=r"$C_L - \frac{(\bar{C_L} - 0)}{\bar{\alpha} -\alpha_{0}} \cdot (x-{\alpha_{0}})$")

axes[1].scatter(180/np.pi*dmean,Z-regalter(a_mw)[0],alpha=alpha*10,c='black')

axes[1].plot(180/np.pi*dmean,zfd_red,
              label=r"Optim: y = %.2f x + %.2f "%(chireg[0],chireg[1])
              ,color="red")

# chi_hand=-0.2*(0.2+0.004)/(np.pi/180*(1.487-(-1.257)))

# axes[1].plot(180/np.pi*dmean,chi_hand*dmean,
#               label=r"Heuristique: y = %.2f x "%(chi_hand)
#               ,color="blue")

axes[1].scatter(180/np.pi*np.mean(dmean),np.mean(Z-regalter(a_mw)[0]),
                c="green",marker="s",s=80.0)

axes[1].legend(),axes[1].set_xlabel(r'$\delta$ (°)'),axes[1].set_ylabel('Cl'),axes[1].grid()


# # input('F')
# # polars

f,ax2es=plt.subplots(2,2)
axes=ax2es.flatten()
atest=np.linspace(-np.pi/180*10.0,np.pi/180*15,100)
deltastest=np.linspace(-15,15,7)*np.pi/180

cl1sa_reg=regalter(a_mw)[1]
chil_reg=chireg[0]/cl1sa_reg

from matplotlib import cm


for i,delta_val in enumerate(deltastest):
    ncolor=cm.viridis(i/len(deltastest))
    
    ls=":" if delta_val!=0 else '-'
    alph=0.5 if delta_val!=0 else 1.0
    axes[1].plot(180.0/np.pi*atest,
                  cdsa0reg+cd1sareg*np.sin(atest-a0reg+chil_reg*delta_val)**2,
                  label=r"$\delta = %.0f $"%(delta_val*180/np.pi),
                  c=ncolor,linestyle=ls,alpha=alph)
        
    axes[0].plot(180.0/np.pi*atest,
                  0.5*cl1sa_reg*np.sin(2*(atest-a0reg)+chil_reg*delta_val),
                  label=r"$\delta = %.0f $"%(delta_val*180/np.pi),
                  linestyle=ls,c=ncolor,alpha=alph)
        
    axes[2].plot(cdsa0reg+cd1sareg*np.sin(atest-a0reg+chil_reg*delta_val)**2,
                  0.5*cl1sa_reg*np.sin(2*(atest-a0reg)+chil_reg*delta_val),
                  label=r"$\delta = %.0f $"%(delta_val*180/np.pi),
                  linestyle=ls,c=ncolor,alpha=alph)
    
    axes[3].plot(180.0/np.pi*atest,
                  0.5*cl1sa_reg*np.sin(2*(atest-a0reg)+chil_reg*delta_val)/(cdsa0reg+cd1sareg*np.sin(atest-a0reg+chil_reg*delta_val)**2),
                  label=r"$\delta = %.0f $"%(delta_val*180/np.pi),
                  linestyle=ls,c=ncolor,alpha=alph)
    
axes[0].grid(),axes[0].set_xlabel(r'$\alpha$ (deg)'),axes[0].set_title(r"$C_L$" ),axes[0].set_ylabel(r'$C_L$ ')

axes[1].grid(),axes[1].legend(),axes[1].set_xlabel(r'$\alpha$ (deg)'),axes[1].set_title(r"$C_D$" ),axes[1].set_ylabel(r'$C_D$ ')

axes[2].grid(),axes[2].set_xlabel(r'$C_D$ '),axes[2].set_ylabel(r'$C_L$ '),axes[2].set_title(r"$C_L vs C_D$" )

axes[3].grid(),axes[3].set_xlabel(r'$\alpha$ (deg)'),axes[3].set_ylabel(r'$C_L/C_D$ '),axes[2].set_title(r"$C_L/C_D$" )



# acc pred 
# acc_log=np.array([df['acc_ned_grad[%i]'%(i)] for i in range(3)]).T

f,axes=plt.subplots(2,1)

aireg=df['gamma[0]']+N2*(cdsa0reg+cd1sareg*np.sin(a_mw-a0reg+chil_reg*dmean)**2)/8.8+0.57*Tint/8.8
akreg=df['gamma[2]']+N3*0.5*cl1sa_reg*np.sin(2*(a_mw-a0reg)+chil_reg*dmean)/8.8
axes=axes.flatten()

print('RMS : %f ai %f ak '%(np.sqrt(np.mean((aireg-df['acc_body_grad[0]'])**2)),
                            np.sqrt(np.mean((akreg-df['acc_body_grad[2]'])**2))))

# axes[0].plot(df['t'],aireg,color="red",label="prediction")
# axes[0].plot(df['t'],df['acc_body_grad[0]'],color="black",label="data")
# axes[0].grid(),axes[0].legend(),axes[0].set_xlabel('$t$'),axes[0].set_ylabel('$a_i$')

# axes[2].plot(Tint,color="black",label="Tint"),axes[2].grid(),axes[2].legend()

# axes[2].plot(df['t'],akreg,color="red",label="prediction")
# axes[2].plot(df['t'],df['acc_body_grad[2]'],color="black",label="data")
# axes[2].grid(),axes[2].legend(),axes[2].set_xlabel('$t$'),axes[2].set_ylabel('$a_k$')


print('RMS : %f ai %f ak '%(np.sqrt(np.mean((aireg-df['acc_body_grad[0]'])**2)),
                            np.sqrt(np.mean((akreg-df['acc_body_grad[2]'])**2))))

ind=(df['t']>50)*(df['t']<220)
axes[0].plot(df['t'][ind],aireg[ind],color="red",label="prediction")
axes[0].plot(df['t'][ind],df['acc_body_grad[0]'][ind],color="black",label="data")
axes[0].grid(),axes[0].legend(),axes[0].set_xlabel('$t$'),axes[0].set_ylabel('$a_i$')

# axes[2].plot(Tint,color="black",label="Tint"),axes[2].grid(),axes[2].legend()

axes[1].plot(df['t'][ind],akreg[ind],color="red",label="prediction")
axes[1].plot(df['t'][ind],df['acc_body_grad[2]'][ind],color="black",label="data")
axes[1].grid(),axes[1].legend(),axes[1].set_xlabel('$t$'),axes[1].set_ylabel('$a_k$')


# axes[3].plot(dmean,color="black",label="dmean"),axes[3].grid(),axes[3].legend()
print("RMS ",np.mean(np.sqrt((aireg-df['acc_body_grad[0]'])**2+(akreg-df['acc_body_grad[2]'])**2)))

# error analysis
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
f,axes=plt.subplots(2,1)

x=df['t']
y=akreg-df['acc_body_grad[2]']
dydx = abs(a_mw)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = axes[0].add_collection(lc)

axes[0].set_xlim(min(x),max(x)),axes[0].set_ylim(min(y),max(y))
axes[0].grid(),axes[0].set_xlabel('$t$'),axes[0].set_ylabel('Error $a_k$')
cbar=f.colorbar(line, ax=axes)
cbar.set_label(r'$|\alpha|$', rotation=0, fontsize=15)
axes[0].set_title("Error vs time")


axes[1].scatter(a_mw,y,alpha=0.005,c=a_mw,cmap='jet')
axes[1].grid(),axes[1].set_xlabel(r'$\alpha$'),axes[1].set_ylabel(r'$a_{k,pred}-a_{k,data}$')
axes[1].set_title(r"Error vs $\alpha$")
# axes[2].plot(Tint,color="black",label="Tint"),axes[2].grid(),axes[2].legend()
input('a')
# PCA 

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(np.c_[a_mw,Z-np.mean(Z)])
f,axes=plt.subplots(1,1)
axes.scatter(a_mw*180/np.pi,Z-np.mean(Z),color="black",alpha=0.005)
axes.grid()
v1,v2=pca.components_.T
axes.quiver(*tuple(v2),color="lightgreen",label="Main Covariance Axis")
axes.quiver(*tuple(v1),color="red",label="Secondary Covariance Axis")

axes.set_xlabel(r'$\alpha °$'),axes.set_ylabel(r'$C_L$'),axes.legend()


print(pca.components_)
print(pca.explained_variance_ratio_)
# acc_log=np.array([df['acc_ned_grad[%i]'%(i)] for i in range(3)]).T

input('F')
# %% usefuncs

# df=pd.read_csv('./id_avion_nosympy_data.csv').drop(columns=["Unnamed: 0"])
# print(df['alphas'].shape)
# input("WELL...")
# 


bounds_ct= (0,1) #1.1e-4/4 1.1e-4*4
bounds_ct10 = (0,1)
bounds_ct20=(None,None)
bounds_a_0= (-0.18,0.18) #15 deg
bounds_a_s= (-np.pi/4,np.pi/4)  #2 30 deg
bounds_d_s= (-np.pi/4,np.pi/4) #2 45 degres
bounds_cl1sa =(0,None)
bounds_cl1fp =(0,None)
bounds_cd1fp =(0,None)
bounds_k0 =(None,None)
bounds_k1 =(None,None)
bounds_k2 =(None,None)
bounds_cd0fp =(0,None)
bounds_cd0sa =(0,None)
bounds_cd1sa= (0,None)
bounds_mass=(5,15)

# %% OPTI INTERM
acc_log=np.array([df['acc[%i]'%(i)] for i in range(3)]).T

alpha_0=-0.077
alpha_s = 0.3391428111
delta_s = 15.0*np.pi/180
cd0sa_0 = 0.02
cd0fp_0 = 0.02
cd1sa_0 = 0.34
cl1sa_0 = 0.5 
cd1fp_0 = 0 
coeff_drag_shift_0= 0.0
coeff_lift_shift_0= 0.05  if not( is_sim_log) else 0.5
coeff_lift_gain_0= -7.5 if not( is_sim_log) else 0.5
C_t0 = 3.2726210849999994e-05 if not( is_sim_log) else 2.5e-5/2


ct = C_t0

ct10= C_t0
ct20 = 0
a_0 =  alpha_0
a_s =  0.3391
d_s =  15.0*np.pi/180
cl1sa = cd1sa_0
cd1fp = cd1fp_0
k0 = 0.0
k1 = 0.0
k2 = coeff_lift_gain_0
cd0fp =  0.0
cd0sa = 0.02
cd1sa = 0.34
m= 8.8 if not( is_sim_log) else 2.5
mref= 8.8 if not( is_sim_log) else 2.5

bounds_ct= (0,1) #1.1e-4/4 1.1e-4*4
bounds_ct10 = (0,1)
bounds_ct20=(None,None)
bounds_a_0= (-0.20,0.20) #7 deg
bounds_a_s= (-np.pi/4,np.pi/4)  #2 30 deg
bounds_d_s= (-np.pi/4,np.pi/4) #2 45 degres
bounds_cl1sa =(0,None)
bounds_cl1fp =(0,None)
bounds_cd1fp =(0,None)
bounds_k0 =(None,None)
bounds_k1 =(None,None)
bounds_k2 =(None,None)
bounds_cd0fp =(0,None)
bounds_cd0sa =(0,None)
bounds_cd1sa= (0,None)
bounds_mass=(5,15)



coeffs_interm_0=np.array([ct10,a_0,
                   cl1sa,k1,k1,k2, 
                   cd0sa, cd1sa,m])

bounds_interm=[bounds_ct,bounds_a_0,bounds_cl1sa,
              bounds_k1,bounds_k1,bounds_k2,bounds_cd0sa,
              bounds_cd1sa,bounds_mass]

def dyn_interm(df=df,coeffs=coeffs_interm_0,
               fix_mass=False,fix_ct=False,no_a_0=False,
               no_k1=False,no_k2=False,no_k0=False):
    
    ct, a_0,cl1sa,k0,k1, k2 , cd0sa, cd1sa,m=coeffs
    
    
    a_0= alpha_0 if no_a_0 else a_0
    m= mref if fix_mass else m
    
    k0 = 0 if no_k0 else k0
    k1 = 0 if no_k1 else k1
    k2 = 0 if no_k2 else k2
    
    
    "compute aero coeffs "
    
    a=np.array([i for i in df['alphas']])
    d_0=np.array([i for i in df['deltas']])
    a_0_arr=a_0*np.ones(a.shape)
    a_0_arr[:,:,-2:]*=0
    
    CL_sa = 1/2 * cl1sa * np.sin(2*(a  - a_0_arr + k1* d_0 ))
    CD_sa = cd0sa + cd1sa * np.sin((a - a_0_arr +k0*d_0))**2


    
    C_L = CL_sa  + k2 * np.sin(d_0)
    C_D = CD_sa 

    #C_L,C_D shape is (n_samples,1,n_surfaces)
    
    # lifts,drags
    ld,dd=np.array([i for i in df['liftdirs']]),np.array([i for i in df['dragdirs']])


    lifts=C_L*ld    
    drags=C_D*dd

    aeroforce_total=np.sum(lifts+drags,axis=2)

    
    # "compute thrust  "

    T=ct*np.array([i for i in df['thrust_dir_ned']]) if not(fix_ct) else 0.57*np.array([i for i in df['Thrust_Reg']])

    g=np.zeros(aeroforce_total.shape)
    g[:,-1]+=9.81
    # print(g)
    forces_total=T+aeroforce_total+m*g
    acc=forces_total/m
    
    return acc



def cost_interm(X,fm=False,fct=False,
                scaling=True,no_a_0=False,
                no_k1=False,no_k2=False,no_k0=False):
    
    X0=X*coeffs_interm_0 if scaling else X
    
    acc=dyn_interm(df,X0,
                   fix_mass=fm,fix_ct=fct,
                   no_a_0=no_a_0,no_k1=no_k1,no_k2=no_k2,no_k0=no_k0)
    
    # ci=np.mean((acc[:,0]-df['acc_ned_grad[0]'])**2,axis=0)/max(abs(df['acc_ned_grad[0]']))**2
    # cj=np.mean((acc[:,1]-df['acc_ned_grad[1]'])**2,axis=0)/max(abs(df['acc_ned_grad[1]']))**2
    # ck=np.mean((acc[:,2]-df['acc_ned_grad[2]'])**2,axis=0)/max(abs(df['acc_ned_grad[2]']))**2
    
    # c=ci+cj+ck
    
    c=np.mean(np.linalg.norm((acc-acc_log),axis=1))

    
    str_top_print="\r "
    for i in X:
        str_top_print=str_top_print+str(round(i,ndigits=5))+" |"
    str_top_print=str_top_print+" "+str(round(c,ndigits=5))
    
    res={}
    l="ct,a_0,cl1sa,k0,k1,k2,cd0sa,cd1sa,m"
    for i,j in zip(l.split(","),X0):
        res[i]=round(j,ndigits=5)       
    res['cost']=c
    # print(res)
    return c

def run_parallel_interm(x):
        fm,fc,scaling,no_a_0,bnds,nok1,nok2,nok0=x
        
        if scaling:
            sol=scipy.optimize.minimize(cost_interm,np.ones(len(coeffs_interm_0)),
            args=(fm,fc,scaling,no_a_0,nok1,nok2,nok0))
        elif bnds:
            sol=scipy.optimize.minimize(cost_interm,coeffs_interm_0,
            args=(fm,fc,scaling,no_a_0,nok1,nok2,nok0),bounds=bounds_interm)
        else:
            sol=scipy.optimize.minimize(cost_interm,coeffs_interm_0,
            args=(fm,fc,scaling,no_a_0,nok1,nok2,nok0))
        
        filename="INTERM_fm_"+str(fm)
        filename=filename+"_fc_"+str(fc)
        filename=filename+"_scaling_"+str(scaling)
        filename=filename+"_bounds_"+str(bnds)
        filename=filename+"_noa0_"+str(no_a_0)
        filename=filename+"_nok1_"+str(nok1)
        filename=filename+"_nok2_"+str(nok2)
        filename=filename+"_nok0_"+str(nok2)

        
        sfile=os.path.join(save_dir,'%s.csv'%(filename))
        keys='cost,fm,fc,scaling,no_a_0,bnds,nok1,nok2,nok0,ct,a_0,cl1sa,k0,k1,k2,cd0sa,cd1sa,m'.replace(' ','').split(",")

        data=[sol['fun']]
        data=data+list(x)
        data=data+(coeffs_interm_0*sol['x']).tolist() if scaling else data+sol['x'].tolist()

        df_save=pd.DataFrame(data=[data],columns=keys)
        df_save.to_csv(sfile)


        return
    

if __name__ == '__main__':

   


    fm_range=[False,True]
    fc_range=[False,True]
    sc_range=[False]
    noa0_range=[True,False]
    bnds_range=[True,False]
    nok1_range=[True,False]
    nok2_range=[True,False]
    nok0_range=[True,False]

    
    x_r=[]
    for j in  fm_range :
        for k in fc_range:
            for l in sc_range:
                for h in noa0_range:
                    for p in bnds_range:
                        for mm in nok1_range:
                            for nn in nok2_range:
                                for ll in nok0_range:
                                    x_r.append([j,k,l,h,p,mm,nn,ll])


            
  
    # pool = Pool(processes=12)
    # pool.map(run_parallel_interm, x_r)

# for i in x_r:
#     run_parallel_interm(i)
    


# %% OPTI SIMPLE

alpha_0=-0.07729
alpha_s = 0.3391428111
delta_s = 15.0*np.pi/180
cd0sa_0 = 0.9
cd0fp_0 = 0.010
cd1sa_0 = 2
cl1sa_0 = 5 
cd1fp_0 = 2.5 
coeff_drag_shift_0= 0.5 
coeff_lift_shift_0= 0.05  if not( is_sim_log) else 0.5
coeff_lift_gain_0= 2.5 if not( is_sim_log) else 0.5
C_t0 = 3.2726210849999994e-05 if not( is_sim_log) else 2.5e-5/2
C_q = 1e-8
C_h = 1e-4

ct = 3.2726210849999994e-05 if not( is_sim_log) else 2.5e-5/2

ct10= 3.2726210849999994e-05
ct20 = 0
a_0 =  0.0
a_s =  0.3391
d_s =  15.0*np.pi/180
cl1sa = 0.7
cd1fp = 0.0
k0 = 0.0
k1 = 0.0
k2 = 0.0
cd0fp =  0.03
cd0sa = 0.03
cd1sa = 0.0
m= 8.8 if not( is_sim_log) else 2.5
mref= 8.8 if not( is_sim_log) else 2.5

coeffs_0=np.array([    ct,\
    a_0, cl1sa,  k0, k1, k2,   cd0sa, cd1sa,\
    a_0,  cl1sa,  k0, k1, k2, cd0sa, cd1sa,  \
    m])


bounds=[bounds_ct,
        bounds_a_0,
        bounds_cl1sa,
        bounds_k0,
        bounds_k1,
        bounds_k2,
        bounds_cd0sa,
        bounds_cd1sa,
        bounds_a_0,
        bounds_cl1sa,
        bounds_k0,
        bounds_k1,
        bounds_k2,
        bounds_cd0sa,
        bounds_cd1sa,
        bounds_mass]

def sigma(a,a_0,a_s,d_s):
    

    s = np.where(abs(a+a_0)>a_s+d_s,
                 np.zeros(len(np.array([a_0]))),
                 np.ones(len(np.array([a_0]))))
    s = np.where(abs(a+a_0)<a_s,
                 np.ones(len(np.array([a_0]))),
                s)
    s = np.where((abs(a+a_0)>a_s)*(abs(a+a_0)<a_s+d_s),
                0.5*(1+np.cos(np.pi*(a+a_0-np.sign(a+a_0)*alpha_s)/d_s)),s)
    return s
    
def dyn(df=df,coeffs=coeffs_0,fix_mass=False,fix_ct=False,
        no_a0=False,no_k1=False,no_k2=False,no_k0=False):
    

    ct,\
    a_0, cl1sa,  k0, k1, k2,   cd0sa, cd1sa,\
    a_0_v,  cl1sa_v,  k0_v, k1_v, k2_v, cd0sa_v, cd1sa_v,  \
    m=coeffs
    
    m = mref if fix_mass else m
    a_0 = alpha_0 if no_a0 else a_0
    "compute aero coeffs "
    k1 = 0 if no_k1 else k1
    k2 = 0 if no_k2 else k2
    k0 = 0 if no_k0 else k0
    k1_v = 0 if no_k1 else k1_v
    k2_v = 0 if no_k2 else k2_v
    k0_v = 0 if no_k0 else k0_v


    a=np.array([i for i in df['alphas']])
    d_0=np.array([i for i in df['deltas']])

    a_0_arr=np.ones(d_0.shape)@np.diag([a_0,a_0,a_0,a_0_v,a_0_v])

    k0d0=d_0@np.diag([k0,k0,k0,k0_v,k0_v])
    k1d0=d_0@np.diag([k1,k1,k1,k1_v,k1_v])

    
    
    CL_sa = 1/2  * np.sin(2*(a + (k1d0) - a_0_arr)) @ np.diag([cl1sa,
                                                           cl1sa,
                                                           cl1sa,
                                                           cl1sa_v,
                                                           cl1sa_v])
    
    CD_sa = np.ones(a.shape)@ np.diag([cd0sa,
                                        cd0sa,
                                        cd0sa,
                                        cd0sa_v,
                                        cd0sa_v])
    
    CD_sa = CD_sa + np.sin((a + (k0d0) - a_0_arr))**2 @ np.diag([cd1sa,
                                                           cd1sa,
                                                           cd1sa,
                                                           cd1sa_v,
                                                           cd1sa_v])
    
    C_L = CL_sa + np.sin(d_0)@np.diag([k2,k2,k2,k2_v,k2_v])
    C_D = CD_sa
   

    
    #C_L,C_D shape is (n_samples,1,n_surfaces)
    
    # lifts,drags
    ld,dd=np.array([i for i in df['liftdirs']]),np.array([i for i in df['dragdirs']])
    
    lifts=C_L*ld    
    drags=C_D*dd
    
    aeroforce_total=np.sum(lifts+drags,axis=2)
    
    # "compute thrust  "

    T=ct*np.array([i for i in df['thrust_dir_ned']]) if not(fix_ct) else 0.57*np.array([i for i in df['Thrust_Reg']])
    g=np.zeros(aeroforce_total.shape)
    g[:,-1]+=9.81
    forces_total=T+aeroforce_total+m*g
    acc=forces_total/m
    
    return acc


def cost(X,fm=False,fct=False,scaling=True,
         no_a_0=False,nok1=False,nok2=False,nok0=False):
    
    X0=X*coeffs_0 if scaling else X
    
    acc=dyn(df,X0,fix_mass=fm,fix_ct=fct,no_a0=no_a_0,
            no_k1=nok1,no_k2=nok2,no_k0=nok0)
    
    
    c=np.mean(np.linalg.norm((acc-acc_log),axis=1))

    print("cost: %f "%(c))    
    return c


def run_parallel_simple(x):
        fm,fc,scaling,no_a_0,bnds,nok1,nok2,nok0=x
        
        if scaling:
            sol=scipy.optimize.minimize(cost,np.ones(len(coeffs_0)),
            args=(fm,fc,scaling,no_a_0,nok1,nok2,nok0))
        elif bnds:
            sol=scipy.optimize.minimize(cost,coeffs_0,
            args=(fm,fc,scaling,no_a_0,nok1,nok2,nok0),bounds=bounds)
        else:
            sol=scipy.optimize.minimize(cost,coeffs_0,
            args=(fm,fc,scaling,no_a_0,nok1,nok2,nok0))
        
        filename="SIMPLE_fm_"+str(fm)
        filename=filename+"_fc_"+str(fc)
        filename=filename+"_scaling_"+str(scaling)
        filename=filename+"_bounds_"+str(bnds)
        filename=filename+"_noa0_"+str(no_a_0)
        filename=filename+"_nok1_"+str(nok1)
        filename=filename+"_nok2_"+str(nok2)
        filename=filename+"_nok0_"+str(nok0)

        sfile=os.path.join(save_dir,'%s.csv'%(filename))
        
        keys='cost,fm,fc,scaling,no_a_0,bnds,nok1,nok2,nok0,ct,\
        a_0, cl1sa,  k0, k1, k2,   cd0sa, cd1sa,\
        a_0_v,  cl1sa_v,  k0_v, k1_v, k2_v, cd0sa_v, cd1sa_v,  \
        m'.replace(' ','').split(",")

        data=[sol['fun']]
        data=data+list(x)
        data=data+(coeffs_0*sol['x']).tolist() if scaling else data+sol['x'].tolist()
        
        df_save=pd.DataFrame(data=[data],columns=keys)
        df_save.to_csv(sfile)

        return
    

if __name__ == '__main__':

    

    fm_range=[False,True]
    fc_range=[False,True]
    sc_range=[False]
    noa0_range=[True,False]
    bnds_range=[True,False]
    nok1_range=[True,False]
    nok2_range=[True,False]
    nok0_range=[True,False]

    x_r=[]
    for j in  fm_range :
        for k in fc_range:
            for l in sc_range:
                for h in noa0_range:
                    for p in bnds_range:
                        for mm in nok1_range:
                            for nn in nok2_range:
                                for ll in nok0_range:
                                    x_r.append([j,k,l,h,p,mm,nn,ll])

            
  
    # pool = Pool(processes=12)
    # pool.map(run_parallel_simple, x_r)


# run(int(input('LAUNCH ? ... \n >>>>')))

# for i in x_r:
#     run_parallel_simple(i)



print('DONE!')


# %% OPTI MULTICOEFFS


alpha_0=-0.07729
alpha_s = 0.3391428111
delta_s = 15.0*np.pi/180
cd0sa_0 = 0.9
cd0fp_0 = 0.010
cd1sa_0 = 2
cl1sa_0 = 5 
cd1fp_0 = 2.5 
coeff_drag_shift_0= 0.5 
coeff_lift_shift_0= 0.05 if not( is_sim_log) else 0.5
coeff_lift_gain_0= 2.5 if not( is_sim_log) else 0.5
C_t0 = 3.2726210849999994e-05 if not( is_sim_log) else 2.5e-5/2
C_q = 1e-8
C_h = 1e-4

ct = 3.2726210849999994e-05 if not( is_sim_log) else 2.5e-5/2
a_0 =  0.15
a_s =  0.3391
d_s =  15.0*np.pi/180
cl1sa = 0.0
cd1fp = 0.0
k0 = 0.0
k1 = 0.0
k2 = 0.0
cd0fp =  0
cs= 0.0
cl1fp=0
cd0sa = 0.0
cd1sa = 0.0


alpha_0=-0.07729
alpha_s = 0.3391428111
delta_s = 15.0*np.pi/180
cd0sa_0 = 0.02
cd0fp_0 = 0.02
cd1sa_0 = 0.34
cl1sa_0 = 0.5 
cd1fp_0 = 0 
coeff_drag_shift_0= 0.0
coeff_lift_shift_0= 0.05  if not( is_sim_log) else 0.5
coeff_lift_gain_0= -7.5 if not( is_sim_log) else 0.5
C_t0 = 3.2726210849999994e-05 if not( is_sim_log) else 2.5e-5/2


ct = C_t0

ct10= C_t0
ct20 = 0
a_0 =  alpha_0
a_s =  0.3391
d_s =  15.0*np.pi/180
cl1sa = cd1sa_0
cd1fp = cd1fp_0
k0 = 0.0
k1 = 0.0
k2 = coeff_lift_gain_0
cd0fp =  0.0
cd0sa = 0.02
cd1sa = 0.34
m= 8.8 if not( is_sim_log) else 2.5
mref= 8.8 if not( is_sim_log) else 2.5



m= 8.8 if not( is_sim_log) else 2.5

coeffs_0_complex=np.array([ct,
                   a_0,
                   a_s,
                   d_s, 
                   cl1sa, 
                   cl1fp,
                   k0, k1, k2, 
                   cs,
                   cd0fp, cd0sa, 
                   cd1sa, cd1fp,
                   a_0,
                   a_s,
                   d_s, 
                   cl1sa, 
                   cl1fp,
                   k0, k1, k2,
                   cs,
                   cd0fp, cd0sa, 
                   cd1sa, cd1fp,
                   m])

def dyn_complex(df=df,coeffs=coeffs_0_complex,
                fix_mass=False,fix_ct=False,
                no_slip=False,no_a_0=False,
                no_k1=False,no_k2=False):
    
    ct,\
    a_0, a_s, d_s, cl1sa, cl1fp, k0, k1, k2, cs, cd0fp, cd0sa, cd1sa, cd1fp, \
    a_0_v, a_s_v, d_s_v, cl1sa_v, cl1fp_v, k0_v, k1_v, k2_v, cs_v, cd0fp_v, cd0sa_v, cd1sa_v, cd1fp_v, \
    m=coeffs
    
    m = mref if fix_mass else m
    a_0= alpha_0 if no_a_0 else a_0
    a_0_v= 0 if no_a_0 else a_0_v
    
    k1 = 0 if no_k1 else k1
    k2 = 0 if no_k2 else k2
    k1_v = 0 if no_k1 else k1_v
    k2_v = 0 if no_k2 else k2_v
    "compute aero coeffs "
    
    a=np.array([i for i in df['alphas']])
    sideslip=np.array([i for i in df['sideslips']])
    d_0=np.array([i for i in df['deltas']])
    
    a_0_arr=np.ones(d_0.shape)@np.diag([a_0,a_0,a_0,a_0_v,a_0_v])

    k0d0=d_0@np.diag([k0,k0,k0,k0_v,k0_v])
    k1d0=d_0@np.diag([k1,k1,k1,k1_v,k1_v])

    
    
    CL_sa = 1/2  * np.sin(2*(a + (k1d0) - a_0_arr)) @ np.diag([cl1sa,
                                                           cl1sa,
                                                           cl1sa,
                                                           cl1sa_v,
                                                           cl1sa_v])
    
    CD_sa = np.ones(a.shape)@ np.diag([cd0sa,
                                        cd0sa,
                                        cd0sa,
                                        cd0sa_v,
                                        cd0sa_v])
    
    CD_sa = CD_sa + np.sin((a + (k0d0) - a_0_arr))**2 @ np.diag([cd1sa,
                                                           cd1sa,
                                                           cd1sa,
                                                           cd1sa_v,
                                                           cd1sa_v])
    
    
    

    CL_fp = 1/2  * np.sin(2*(a + (k1d0) - a_0_arr)) @ np.diag([cl1fp,
                                                           cl1fp,
                                                           cl1fp,
                                                           cl1fp_v,
                                                           cl1fp_v])
    
    CD_fp = np.ones(a.shape)@ np.diag([cd0fp,
                                        cd0fp,
                                        cd0fp,
                                        cd0fp_v,
                                        cd0fp_v])
    
    CD_fp = CD_fp + np.sin((a + (k0d0) - a_0_arr))**2 @ np.diag([cd1fp,
                                                           cd1fp,
                                                           cd1fp,
                                                           cd1fp_v,
                                                           cd1fp_v])
    
    

    # puiss=5
    # s = - ((a+a_0)**2 @(np.diag(1.0/np.array([a_s,
    #                                           a_s,
    #                                           a_s,
    #                                           a_s_v,
    #                                           a_s_v])))**2)**puiss
    # s = s @ (((a+a_0)**2@(np.diag(1.0/np.array([a_s,
    #                                             a_s,
    #                                             a_s,
    #                                             a_s_v,
    #                                             a_s_v])))**2)**puiss+ 100+200* np.diag([ d_s,
    #                                                                                     d_s,
    #                                                                                     d_s
    #                                                                                     ,d_s_v,
    #                                                                                     d_s_v]))
    # s = s+1
    s=sigma(a,a_0,
            np.array([a_s,a_s,a_s,a_s_v,a_s_v]),
            np.array([d_s,d_s,d_s,d_s_v,d_s_v]))


    C_L = CL_fp + s*(CL_sa - CL_fp) 
    C_L = C_L + np.sin(d_0)@np.diag([k2,k2,k2,k2_v,k2_v])
    C_D = CD_fp + s*(CD_sa - CD_fp)
    C_S =np.sin(sideslip)@np.diag([cs,cs,cs,cs_v,cs_v])
    #C_L,C_D shape is (n_samples,1,n_surfaces)
    
    # lifts,drags
    ld,dd=np.array([i for i in df['liftdirs']]),np.array([i for i in df['dragdirs']])
    sd=np.array([i for i in df['sideslips']])
    
    lifts=C_L*ld    
    drags=C_D*dd
    sweep=C_S*sd
    # aeroforce_total=np.sum(lifts+drags,axis=2)
    aeroforce_total=np.sum(lifts+drags,axis=2)  if no_slip else  np.sum(lifts+drags+sweep,axis=2) 
    # "compute thrust  "

    T=ct*np.array([i for i in df['thrust_dir_ned']]) if not(fix_ct) else 0.57*np.array([i for i in df['Thrust_Reg']])
    g=np.zeros(aeroforce_total.shape)
    g[:,-1]+=9.81
    forces_total=T+aeroforce_total+m*g
    acc=forces_total/m
    
    return acc


acc_log=np.array([df['acc[%i]'%(i)] for i in range(3)]).T

def cost_ext(X,
             fm=False,fct=False,
             no_slip=False,scaling=True,
             no_a_0=False,no_k1=False,no_k2=False,verbose=True):
    
    X0=X*coeffs_0_complex if scaling else X

    acc=dyn_complex(df,X0,
                    fix_mass=fm,fix_ct=fct,
                    no_a_0=no_a_0,no_slip=no_slip,
                    no_k1=no_k1,no_k2=no_k2)
    
    
    c=np.mean(np.linalg.norm((acc-acc_log),axis=1))

    res={}
    res['cost']=c
    print(res) if verbose else None
    return c

bboudns=[bounds_ct,
        bounds_a_0,
        bounds_a_s,
        bounds_d_s,
        bounds_cl1sa,
        bounds_cd1fp,
        bounds_k0,
        bounds_k1,
        bounds_k2,
        bounds_k2, #cs
        bounds_cd0fp,
        bounds_cd0sa,
        bounds_cd1sa,
        bounds_cd1fp,
        bounds_mass]
bounds_complex=[i for i in bboudns[:-1]]+[i for i in bboudns[1:]]

def run_parallel_complex(x):
        fm,fc,sideslip,scaling,no_a_0,bnds,nok1,nok2=x
    
        if scaling:
            sol=scipy.optimize.minimize(cost_ext,np.ones(len(coeffs_0_complex)),
            args=(fm,fc,sideslip,scaling,no_a_0,nok1,nok2))
        elif bnds:
            sol=scipy.optimize.minimize(cost_ext,coeffs_0_complex,
            args=(fm,fc,sideslip,scaling,no_a_0,nok1,nok2),bounds=bounds_complex)  
        else:
            sol=scipy.optimize.minimize(cost_ext,coeffs_0_complex,
            args=(fm,fc,sideslip,scaling,no_a_0,nok1,nok2))
    
        filename="COMPLEX_fm_"+str(fm)
        filename=filename+"_fc_"+str(fc)
        filename=filename+"_sideslip_"+str(sideslip)
        filename=filename+"_scaling_"+str(scaling)
        filename=filename+"_bounds_"+str(bnds)
        filename=filename+"_noa0_"+str(no_a_0)
        filename=filename+"_nok1_"+str(nok1)
        filename=filename+"_nok2_"+str(nok2)
        
        
        sfile=os.path.join(save_dir,'%s.csv'%(filename))

        keys='cost,fm,fc,sideslip,scaling,no_a_0,bnds,nok1,nok2,ct,\
            a_0, a_s, d_s, cl1sa, cl1fp, k0, k1, k2, cs, cd0fp, cd0sa, cd1sa, cd1fp, \
            a_0_v, a_s_v, d_s_v, cl1sa_v, cl1fp_v, k0_v, k1_v, k2_v, cs_v, cd0fp_v, cd0sa_v, cd1sa_v, cd1fp_v, \
            m'.replace(' ','').split(",")

        data=[sol['fun']]
        data=data+list(x)
        data=data+(coeffs_0_complex*sol['x']).tolist() if scaling else data+sol['x'].tolist()
        
        df_save=pd.DataFrame(data=[data],columns=keys)
        df_save.to_csv(sfile)
        return

if __name__ == '__main__':

    

    fm_range=[True]
    fc_range=[True,False]
    sidslip_range=[False]
    sc_range=[False]
    noa0_range=[True,False]
    bnds_range=[True,False]
    nok1_range=[True,False]
    nok2_range=[True,False]
    
    x_r=[]
    for j in  fm_range :
        for k in fc_range:
            for i in sidslip_range:
                for l in sc_range:
                    for h in noa0_range:
                        for p in bnds_range:
                            for mm in nok1_range:
                                for nn in nok2_range:
                                
                                        x_r.append([j,k,i,l,h,p,mm,nn])
                        
  
    # pool = Pool(processes=12)
    # pool.map(run_parallel_complex, x_r)

# for i in x_r:
#     run_parallel_complex(i)

# run(int(input('LAUNCH ? ... \n >>>>')))


# plt.figure()
# x=[i*j for (i,j) in zip(np.array([i for i in alphas])[:,:,0],(df['speed[0]']**2+df['speed[1]']**2+df['speed[2]']**2))]
# y=df['acc_body_grad[0]']-df['thrust_intensity']
# plt.scatter(x,y,alpha=0.01)


acc_pred=dyn(df=df,coeffs=coeffs_0,fix_mass=False,fix_ct=False,
        no_a0=False,no_k1=False,no_k2=False,no_k0=False)
