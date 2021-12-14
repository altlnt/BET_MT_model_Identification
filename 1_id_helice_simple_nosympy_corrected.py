#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:59:50 2021

@author: alex
"""
import numpy as np

mass=369 #batterie
mass+=1640-114 #corps-carton
mass/=1e3
Area=np.pi*(11.0e-02)**2
r0=11e-02
rho0=1.204
kv_motor=800.0
pwmmin=1075.0
pwmmax=1950.0
U_batt=16.8

b10=14.44

# %%   ####### IMPORT DATA 
print("LOADING DATA...")
import pandas as pd


log_path="./logs/copter/vol12/log_real_processed.csv"

raw_data=pd.read_csv(log_path)

print("PROCESSING DATA...")


prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])
# print(prep_data)

if "vol12" in log_path:
    tmin,tmax=(-1,1e10) 
elif "vol1" in log_path:
    tmin,tmax=(41,265) 
elif "vol2" in log_path:
    tmin,tmax=(10,140) 
    
prep_data=prep_data[prep_data['t']>tmin]
prep_data=prep_data[prep_data['t']<tmax]
prep_data=prep_data.reset_index()
for i in range(3):
    prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
    
    

prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
prep_data['t']-=prep_data['t'][0]
prep_data=prep_data.drop(index=[0,len(prep_data)-1])




for i in range(6):
    prep_data['omega_c[%i]'%(i+1)]=(prep_data['PWM_motor[%i]'%(i+1)]-pwmmin)/(pwmmax-pwmmin)*U_batt*kv_motor*2*np.pi/60


# %%   ####### Identify Thrust 

def compute_single_motor_thrust_MT(c1,vak,omega,c2=0,vanilla_test=False):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)

    T=2*rho0*Area*eta*(eta-vak)

    if vanilla_test:
        T=c1*omega**2
    return T


def compute_single_motor_thrust_BET(c1,vak,omega,c2=0,vanilla_test=False):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)

    T=rho0*Area*r0*omega*(c1*r0*omega-c2*(eta-vak))
    if vanilla_test:
        T=c1*omega**2
    return T

def compute_acc_k(c1,c2=0,df=prep_data,vanilla=False,model="MT"):
    
    vak=df["speed_body[2]"]
    gamma=df["gamma[2]"]
    
    if model=="MT":
        T_sum=sum([compute_single_motor_thrust_MT(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    elif model=="BET":
        T_sum=sum([compute_single_motor_thrust_BET(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    else:
        return print("FIX MODEL")
    acc_k=-T_sum/mass+gamma
    
    return acc_k

from scipy.optimize import minimize

import matplotlib.pyplot as plt


def cost_vanilla(X):
    c1=X
    Y=compute_acc_k(c1,vanilla=True)
    c=np.mean((Y-prep_data['acc_body_grad[2]'])**2,axis=0)
    print("c1 :%f ,c2: VANILLA ,cost :%f"%(c1,c))
    return c

X0_vanilla=np.array([6e-6])

sol_vanilla=minimize(cost_vanilla,X0_vanilla,method="SLSQP")
c1vanilla=sol_vanilla['x']
print("\n \n")

def cost(X):
    c1,c2=X
    Y=compute_acc_k(c1,c2=c2)
    c=np.mean((Y-prep_data['acc_body_grad[2]'])**2,axis=0)
    print("c1 :%f ,c2: %f,cost :%f"%(c1,c2,c))
    return c


X0=np.zeros(2)
sol_custom=minimize(cost,X0,method="SLSQP")

c1sol,c2sol=sol_custom['x']
# %%% Comparison

f=plt.figure()
f.suptitle("No drag")
ax=f.add_subplot(2,1,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[2]'],color="black",label="log")
ax.plot(prep_data["t"],compute_acc_k(c1vanilla,vanilla=True),color="red",label="pred",alpha=0.5)
ax.plot(prep_data["t"],compute_acc_k(c1sol,c2=c2sol,model="MT"),color="blue",label="optimized, MT",alpha=0.5)
ax.plot(prep_data["t"],compute_acc_k(c1sol,c2=c2sol,model="BET"),color="green",label="optimized, MT",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
s="%f for vanilla, %f for custom model"%(sol_vanilla['fun'],sol_custom['fun'])
print(s)
ax.set_title(s)

print('\n\nCoherence with ct2=ct1*b1-2/b1 formula ?\n')
print('with the formula : ')
print("ct2=%f"%(c1sol*b10-2/b10))
print("with the identification : ")
print("ct2=%f"%(c2sol))


print('\n\nCoherence with TMT=TBET ?\n')
yrms=np.sqrt(np.mean((compute_acc_k(c1sol,c2=c2sol,model="MT")-compute_acc_k(c1sol,c2=c2sol,model="BET"))**2))
print("output difference rms : %s m/s"%(yrms))

# %%   ####### Identify Thrust(with dk)


def compute_single_motor_thrust_MT_wdrag(c1,vak,omega,c2=0,vanilla_test=False):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)

    T=2*rho0*Area*eta*(eta-vak)

    if vanilla_test:
        T=c1*omega**2
    return T


def compute_single_motor_thrust_BET_wdrag(c1,vak,omega,c2=0,vanilla_test=False):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)

    T=rho0*Area*r0*omega*(c1*r0*omega-c2*(eta-vak))
    if vanilla_test:
        T=c1*omega**2
    return T

def compute_acc_k_wdrag(c1,dk,c2=0,df=prep_data,vanilla=False,model="MT"):
    
    vak=df["speed_body[2]"]
    gamma=df["gamma[2]"]
    
    if model=="MT":
        T_sum=sum([compute_single_motor_thrust_MT(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    elif model=="BET":
        T_sum=sum([compute_single_motor_thrust_BET(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    else:
        return print("FIX MODEL")
    acc_k=-T_sum/mass+gamma-rho0*Area*dk*np.abs(vak)*vak
    
    return acc_k

from scipy.optimize import minimize

import matplotlib.pyplot as plt


def cost_vanilla_wdrag(X):
    c1,dk=X
    Y=compute_acc_k_wdrag(c1,dk,vanilla=True)
    c=np.mean((Y-prep_data['acc_body_grad[2]'])**2,axis=0)
    print("c1 :%f ,c2: VANILLA , dk: %f ,cost :%f"%(c1,dk,c))
    return c

X0_vanilla=np.array([6e-6,0])

sol_vanilla_drag=minimize(cost_vanilla_wdrag,X0_vanilla,method="SLSQP")
c1vanilla,dkvanilla=sol_vanilla_drag['x']


def cost_wdrag(X):
    c1,c2,dk=X
    Y=compute_acc_k_wdrag(c1,dk,c2=c2)
    c=np.mean((Y-prep_data['acc_body_grad[2]'])**2,axis=0)
    print("c1 :%f ,c2: %f, dk: %f , cost :%f"%(c1,c2,dk,c))
    return c


X0=np.zeros(3)
sol_custom_drag=minimize(cost_wdrag,X0,method="SLSQP")

c1sol,c2sol,dksol=sol_custom_drag['x']

# %%% Comparison

f.suptitle("Thrust no drag / With drag")
ax=f.add_subplot(2,1,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[2]'],color="black",label="log")
ax.plot(prep_data["t"],compute_acc_k_wdrag(c1vanilla,dkvanilla,vanilla=True),color="red",label="pred",alpha=0.5)
ax.plot(prep_data["t"],compute_acc_k_wdrag(c1sol,dksol,c2=c2sol,model="MT"),color="blue",label="optimized, MT",alpha=0.5)
ax.plot(prep_data["t"],compute_acc_k_wdrag(c1sol,dksol,c2=c2sol,model="BET"),color="green",label="optimized, MT",alpha=0.5)
ax.legend()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
s="%f for vanilla, %f for custom model"%(sol_vanilla_drag['fun'],sol_custom_drag['fun'])
ax.set_title(s)
print(s)


print('\n\nCoherence with ct2=ct1*b1-2/b1 formula ?\n')
print('with the formula : ')
print("ct2=%f"%(c1sol*b10-2/b10))
print("with the identification : ")
print("ct2=%f"%(c2sol))


print('\n\nCoherence with TMT=TBET ?\n')
yrms=np.sqrt(np.mean((compute_acc_k_wdrag(c1sol,dksol,c2=c2sol,model="MT")-compute_acc_k_wdrag(c1sol,dksol,c2=c2sol,model="BET"))**2))
print("output difference rms : %s m/s"%(yrms))

# %%% Comparison
f.suptitle("Vanilla / Augmented with drag")
ax=f.add_subplot(2,1,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[2]'],color="black",label="log")
ax.plot(prep_data["t"],compute_acc_k_wdrag(c1vanilla,dkvanilla,vanilla=True),color="darkred",label="pred",alpha=0.5)
ax.plot(prep_data["t"],compute_acc_k_wdrag(c1sol,dksol,c2=c2sol,model="MT"),color="darkblue",label="optimized, MT",alpha=0.5)
ax.plot(prep_data["t"],compute_acc_k_wdrag(c1sol,dksol,c2=c2sol,model="BET"),color="darkgreen",label="optimized, MT",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
print("%f for vanilla, %f for custom model"%(sol_vanilla_drag['fun'],sol_custom_drag['fun']))


print('\n\nCoherence with ct2=ct1*b1-2/b1 formula ?\n')
print('with the formula : ')
print("ct2=%f"%(c1sol*b10-2/b10))
print("with the identification : ")
print("ct2=%f"%(c2sol))


print('\n\nCoherence with TMT=TBET ?\n')
yrms=np.sqrt(np.mean((compute_acc_k_wdrag(c1sol,dkvanilla,c2=c2sol,model="MT")-compute_acc_k_wdrag(c1sol,dkvanilla,c2=c2sol,model="BET"))**2))
print("output difference rms : %s m/s"%(yrms))

# %%% ai



# %% ai
# %%%   ####### Identify pure drag

def compute_ai_od(di,df=prep_data):
    
    vak=df["speed_body[0]"]
    Fa=-rho0*Area*di*np.abs(vak)*vak
    gamma=df["gamma[0]"]

    return Fa+gamma

def cost_ai_onlydrag(X):
    di=X
    
    Y=compute_ai_od(di)
    c=np.mean((Y-prep_data['acc_body_grad[0]'])**2,axis=0)
    print("di :%f , cost :%f"%(di,c))

    return c
    
X0_di_onlydrag=np.array([0])

sol_ai_od=minimize(cost_ai_onlydrag,X0_di_onlydrag,method="SLSQP")
di_only_=sol_ai_od['x']
print("\n \n")

# %%%   ####### Identify H-force nodrag


def compute_eta(vak,omega,c1=c1sol,c2=c2sol):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)
    return eta

def compute_H(vak,omega,ch1,ch2):
    eta=compute_eta(vak,omega)
    H=rho0*Area*(ch1*r0*omega-ch2*(eta-vak))
    return H

def compute_ai_H_only(ch1,ch2,df=prep_data):

    vai=df["speed_body[0]"]
    vak=df["speed_body[2]"]

    gamma=df["gamma[0]"]
    
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    H_vect=-vai*H
    
    return H_vect+gamma

def cost_ai_h_only(X):
    ch1,ch2=X

    Y=compute_ai_H_only(ch1,ch2)
    c=np.mean((Y-prep_data['acc_body_grad[0]'])**2,axis=0)
    print("ch1 :%f , ch2 :%f , cost :%f"%(ch1,ch2,c))

    return c

X0_ai_onlyh=np.array([0,0])

sol_ai_oh=minimize(cost_ai_h_only,X0_ai_onlyh,method="SLSQP")
ch1_ai_only_,ch2_ai_only_=sol_ai_oh['x']
print("\n \n")

# %%%   ####### Identify H-force wdrag


def compute_eta(vak,omega,c1=c1sol,c2=c2sol):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)
    return eta

def compute_H(vak,omega,ch1,ch2):
    eta=compute_eta(vak,omega)
    H=rho0*Area*(ch1*r0*omega-ch2*(eta-vak))
    return H

def compute_ai_H_wdrag(ch1,ch2,di,df=prep_data):
    
    vai=df["speed_body[0]"]
    vak=df["speed_body[2]"]

    gamma=df["gamma[0]"]
    
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    H_vect=-vai*H
    Fa=-rho0*Area*di*np.abs(vai)*vai

    return H_vect+gamma+Fa

def cost_ai_h_wdrag(X):
    ch1,ch2,di=X

    Y=compute_ai_H_wdrag(ch1,ch2,di)
    c=np.mean((Y-prep_data['acc_body_grad[0]'])**2,axis=0)
    print("ch1 :%f , ch2 :%f , di :%f , cost :%f"%(ch1,ch2,di,c))

    return c

X0_ai_hwd=np.array([0,0,0])

sol_ai_hwd=minimize(cost_ai_h_wdrag,X0_ai_hwd,method="SLSQP")
ch1_ai_wd_,ch2_ai_wd_,di_wd_=sol_ai_hwd['x']

# %%%   ####### Comparison

f=plt.figure()
f.suptitle("Ai drag vs H force fit")
ax=f.add_subplot(1,1,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[0]'],color="black",label="log")
ax.plot(prep_data["t"],compute_ai_od(di_only_),color="darkred",label="pure drag",alpha=0.5)
ax.plot(prep_data["t"],compute_ai_H_only(ch1_ai_only_,ch2_ai_only_),color="darkblue",label="pure h force",alpha=0.5)
ax.plot(prep_data["t"],compute_ai_H_wdrag(ch1_ai_only_,ch2_ai_only_,di_wd_),color="darkgreen",label="drag + h force",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
s="%f for vanilla, %f for custom model, %f for full model"%(sol_ai_od['fun'],sol_ai_oh['fun'],sol_ai_hwd['fun'])
ax.set_title(s)
print(s)

# %% aj
# %%%   ####### Identify pure drag

def compute_aj_od(dj,df=prep_data):
    
    vak=df["speed_body[1]"]
    Fa=-rho0*Area*dj*np.abs(vak)*vak
    gamma=df["gamma[1]"]

    return Fa+gamma

def cost_aj_onlydrag(X):
    dj=X
    
    Y=compute_aj_od(dj)
    c=np.mean((Y-prep_data['acc_body_grad[1]'])**2,axis=0)
    print("dj :%f , cost :%f"%(dj,c))

    return c
    
X0_dj_onlydrag=np.array([1])

sol_aj_od=minimize(cost_aj_onlydrag,X0_dj_onlydrag,method="SLSQP")
dj_only_=sol_aj_od['x']
print("\n \n")

# %%%   ####### Identify H-force nodrag


def compute_eta(vak,omega,c1=c1sol,c2=c2sol):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)
    return eta

def compute_H(vak,omega,ch1,ch2):
    eta=compute_eta(vak,omega)
    H=rho0*Area*(ch1*r0*omega-ch2*(eta-vak))
    return H

def compute_aj_H_only(ch1,ch2,df=prep_data):
    
    vak=df["speed_body[2]"]
    vaj=df["speed_body[1]"]

    gamma=df["gamma[1]"]
    
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    H_vect=-vaj*H
    
    return H_vect+gamma

def cost_aj_h_only(X):
    ch1,ch2=X

    Y=compute_aj_H_only(ch1,ch2)
    c=np.mean((Y-prep_data['acc_body_grad[1]'])**2,axis=0)
    print("ch1 :%f , ch2 :%f , cost :%f"%(ch1,ch2,c))

    return c

X0_aj_onlyh=np.array([0,0])

sol_aj_oh=minimize(cost_aj_h_only,X0_aj_onlyh,method="SLSQP")
ch1_aj_only_,ch2_aj_only_=sol_aj_oh['x']
print("\n \n")

# %%%   ####### Identify H-force wdrag


def compute_eta(vak,omega,c1=c1sol,c2=c2sol):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)
    return eta

def compute_H(vak,omega,ch1,ch2):
    eta=compute_eta(vak,omega)
    H=rho0*Area*(ch1*r0*omega-ch2*(eta-vak))
    return H

def compute_aj_H_wdrag(ch1,ch2,dj,df=prep_data):
    
    vak=df["speed_body[2]"]
    vaj=df["speed_body[1]"]

    gamma=df["gamma[1]"]
    
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    H_vect=-vaj*H
    Fa=-rho0*Area*dj*np.abs(vaj)*vaj

    return H_vect+gamma+Fa

def cost_aj_h_wdrag(X):
    ch1,ch2,dj=X

    Y=compute_aj_H_wdrag(ch1,ch2,dj)
    c=np.mean((Y-prep_data['acc_body_grad[1]'])**2,axis=0)
    print("ch1 :%f , ch2 :%f , dj :%f , cost :%f"%(ch1,ch2,dj,c))

    return c

X0_aj_hwd=np.array([0,0,0])

sol_aj_hwd=minimize(cost_aj_h_wdrag,X0_aj_hwd,method="SLSQP")
ch1_aj_wd_,ch2_aj_wd_,dj_wd_=sol_aj_hwd['x']

# %%%   ####### Comparison

f=plt.figure()
f.suptitle("Aj drag vs H force fit")
ax=f.add_subplot(1,1,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[1]'],color="black",label="log")
ax.plot(prep_data["t"],compute_aj_od(dj_only_),color="darkred",label="pure drag",alpha=0.5)
ax.plot(prep_data["t"],compute_aj_H_only(ch1_aj_only_,ch2_aj_only_),color="darkblue",label="pure h force",alpha=0.5)
ax.plot(prep_data["t"],compute_aj_H_wdrag(ch1_aj_only_,ch2_aj_only_,dj_wd_),color="darkgreen",label="drag +h force",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
s="%f for vanilla \n %f for custom model \n %f for full model"%(sol_aj_od['fun'],sol_aj_oh['fun'],sol_aj_hwd["fun"])
ax.set_title(s)
print(s)
# %% aij

# %%% H nodrag

def compute_aij_H_wdrag(ch1,ch2,di=0,dj=0,df=prep_data):
    
    vai=df["speed_body[0]"]
    vaj=df["speed_body[1]"]
    vak=df["speed_body[2]"]
    gammai=df["gamma[0]"]
    gammaj=df["gamma[1]"]

    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    H_vect=np.c_[-vai*H,-vaj*H]
    
    Fa=-rho0*Area*np.c_[di*np.abs(vai)*vai,dj*np.abs(vaj)*vaj]

    return H_vect+np.c_[gammai,gammaj]+Fa

def cost_aij_h_nodrag(X):
    ch1,ch2=X
    Y=compute_aij_H_wdrag(ch1,ch2,di=0,dj=0)
    ci=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2,axis=0)
    cj=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2,axis=0)

    c=ci+cj
    print("ch1 :%f , ch2 :%f , cost :%f"%(ch1,ch2,c))

    return c

X0_aij_nodrag=np.array([0,0])

sol_aij_nodrag=minimize(cost_aij_h_nodrag,X0_aij_nodrag,method="SLSQP")
ch1_aij_nodrag_,ch2_aij_nodrag_=sol_aij_nodrag['x']


# %%% H wd


def cost_aij_h_wdrag(X):
    ch1,ch2,di,dj=X

    Y=compute_aij_H_wdrag(ch1,ch2,di,dj)

    ci=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2,axis=0)
    cj=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2,axis=0)

    c=ci+cj

    print("ch1 :%f , ch2 :%f , di :%f , dj : %f , cost :%f"%(ch1,ch2,di,dj,c))

    return c

X0_aij_hwd=np.array([0,0,0,0])

sol_aij_hwd=minimize(cost_aij_h_wdrag,X0_aij_hwd,method="SLSQP")
ch1_aij_wd_,ch2_aij_wd_,di_aij_wd_,dj_aij_wd_=sol_aij_hwd['x']


# %%% Comparison ai
aind,ajnd=compute_aij_H_wdrag(ch1_aij_nodrag_,ch2_aij_nodrag_).T
aid,ajd=compute_aij_H_wdrag(ch1_aij_wd_,ch2_aij_wd_,di_aij_wd_,dj_aij_wd_).T

f=plt.figure()
f.suptitle("Aij drag vs H force fit, nodrag")
ax=f.add_subplot(1,2,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[0]'],color="black",label="log")
ax.plot(prep_data["t"],compute_ai_od(di_only_),color="darkred",label="pure drag",alpha=0.5)
ax.plot(prep_data["t"],aind,color="darkblue",label="pure h force",alpha=0.5)
ax.plot(prep_data["t"],aid,color="darkgreen",label="drag +h force",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
c_i_nd=np.mean((aind-prep_data['acc_body_grad[0]'])**2,axis=0)
c_i_d=np.mean((aid-prep_data['acc_body_grad[0]'])**2,axis=0)                            
s="%f for vanilla \n  %f for custom model \n %f for full model"%(sol_ai_od['fun'],c_i_nd,c_i_d)
ax.set_title(s)
print(s)

# %%% Comparison aj

ax=f.add_subplot(1,2,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[1]'],color="black",label="log")
ax.plot(prep_data["t"],compute_aj_od(dj_only_),color="darkred",label="pure drag",alpha=0.5)
ax.plot(prep_data["t"],ajnd,color="darkblue",label="pure h force",alpha=0.5)
ax.plot(prep_data["t"],ajd,color="darkgreen",label="drag +h force",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
c_j_nd=np.mean((ajnd-prep_data['acc_body_grad[0]'])**2,axis=0)
c_j_d=np.mean((ajd-prep_data['acc_body_grad[0]'])**2,axis=0)  
s="%f for vanilla \n %f for custom model \n %f for full model"%(sol_aj_od['fun'],c_j_nd,c_j_d)
ax.set_title(s)
print(s)



# %% aij (di_eq_dj)
 
# %%% H nodrag

def compute_aij_H_wdrag(ch1,ch2,di=0,dj=0,df=prep_data):
    
    vai=df["speed_body[0]"]
    vaj=df["speed_body[1]"]
    vak=df["speed_body[2]"]
    gammai=df["gamma[0]"]
    gammaj=df["gamma[1]"]

    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    H_vect=np.c_[-vai*H,-vaj*H]
    
    Fa=-rho0*Area*np.c_[di*np.abs(vai)*vai,dj*np.abs(vaj)*vaj]

    return H_vect+np.c_[gammai,gammaj]+Fa


# %%% H wd


def cost_aij_h_wdrag_di_eq_dj_(X):
    ch1,ch2,di=X

    Y=compute_aij_H_wdrag(ch1,ch2,di,di)

    ci=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2,axis=0)
    cj=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2,axis=0)

    c=ci+cj

    print("ch1 :%f , ch2 :%f , dij :%f  , cost :%f"%(ch1,ch2,di,c))

    return c

X0_aij_hwd_di_eq_dj_=np.array([0,0,0])

sol_aij_hwd_di_eq_dj_=minimize(cost_aij_h_wdrag_di_eq_dj_,X0_aij_hwd_di_eq_dj_,method="SLSQP")
ch1_aij_wd_di_eq_dj_,ch2_aij_wd_di_eq_dj_,dij_aij_wd_di_eq_dj_=sol_aij_hwd_di_eq_dj_['x']


# %%% Comparison ai
aind,ajnd=compute_aij_H_wdrag(ch1_aij_wd_di_eq_dj_,ch2_aij_wd_di_eq_dj_).T
aid,ajd=compute_aij_H_wdrag(ch1_aij_wd_di_eq_dj_,ch2_aij_wd_di_eq_dj_,dij_aij_wd_di_eq_dj_,dij_aij_wd_di_eq_dj_).T

f=plt.figure()
f.suptitle("Aij drag vs H force fit wdrag")
ax=f.add_subplot(1,2,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[0]'],color="black",label="log")
ax.plot(prep_data["t"],compute_ai_od(di_only_),color="darkred",label="pure drag",alpha=0.5)
ax.plot(prep_data["t"],aind,color="darkblue",label="pure h force",alpha=0.5)
ax.plot(prep_data["t"],aid,color="darkgreen",label="drag +h force",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
c_i_nd=np.mean((aind-prep_data['acc_body_grad[0]'])**2,axis=0)
c_i_d=np.mean((aid-prep_data['acc_body_grad[0]'])**2,axis=0)                            
s="%f for vanilla \n %f for custom model \n %f for full model"%(sol_aij_hwd_di_eq_dj_['fun'],c_i_nd,c_i_d)
ax.set_title(s)
print(s)

# %%% Comparison aj

ax=f.add_subplot(1,2,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[1]'],color="black",label="log")
ax.plot(prep_data["t"],compute_aj_od(dj_only_),color="darkred",label="pure drag",alpha=0.5)
ax.plot(prep_data["t"],ajnd,color="darkblue",label="pure h force",alpha=0.5)
ax.plot(prep_data["t"],ajd,color="darkgreen",label="drag +h force",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
c_j_nd=np.mean((ajnd-prep_data['acc_body_grad[0]'])**2,axis=0)
c_j_d=np.mean((ajd-prep_data['acc_body_grad[0]'])**2,axis=0)  
s="%f for vanilla \n %f for custom model \n %f for full model"%(sol_aij_hwd_di_eq_dj_['fun'],c_j_nd,c_j_d)
ax.set_title(s)
print(s)

# %% Global 

def compute_eta(vak,omega,c1=c1sol,c2=c2sol):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)
    return eta

def compute_H(vak,omega,ch1,ch2):
    eta=compute_eta(vak,omega)
    H=rho0*Area*(ch1*r0*omega-ch2*(eta-vak))
    return H


def compute_single_motor_thrust_MT(c1,vak,omega,c2=0,vanilla_test=False):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)

    T=2*rho0*Area*eta*(eta-vak)

    if vanilla_test:
        T=c1*omega**2
    return T

def compute_acc_k(c1,c2=0,df=prep_data,vanilla=False,model="MT"):
    
    vak=df["speed_body[2]"]
    gamma=df["gamma[2]"]
    
    if model=="MT":
        T_sum=sum([compute_single_motor_thrust_MT(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    elif model=="BET":
        T_sum=sum([compute_single_motor_thrust_BET(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    else:
        return print("FIX MODEL")
    acc_k=-T_sum/mass+gamma
    
    return acc_k


def compute_acc_global(ct1,ct2,ch1,ch2,di=0,dj=0,dk=0,df=prep_data):
    
    vai=df["speed_body[0]"]
    vaj=df["speed_body[1]"]
    vak=df["speed_body[2]"]
    
    gammai=df["gamma[0]"]
    gammaj=df["gamma[1]"]
    gammak=df["gamma[2]"]
    
    T=sum([compute_single_motor_thrust_MT(ct1,vak,df['omega_c[%i]'%(i+1)],ct2) for i in range(6)])
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    
    H_vect=np.c_[-vai*H,-vaj*H,np.zeros(H.shape)]
    T_vect=np.c_[np.zeros(T.shape),np.zeros(T.shape),T]
    absva=np.sqrt(vai**2+vaj**2+vak**2)
    Fa=-rho0*Area*np.c_[di*absva*vai,dj*absva*vaj,dk*absva*vak]

    return -T_vect/mass+H_vect+np.c_[gammai,gammaj,gammak]+Fa


def cost_global_(X):
    ct1,ct2,ch1,ch2,di,dj,dk=X

    Y=compute_acc_global(ct1,ct2,ch1,ch2,di,dj,dk)

    ci=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2/max(abs(prep_data['acc_body_grad[0]']))**2,axis=0)
    cj=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2/max(abs(prep_data['acc_body_grad[1]']))**2,axis=0)
    ck=np.mean((Y[:,2]-prep_data['acc_body_grad[2]'])**2/max(abs(prep_data['acc_body_grad[2]']))**2,axis=0)
    
    c=ci+cj+ck

    print("ct1 :%f, ct2 :%f , ch1 :%f , ch2 :%f , di :%f , dj : %f , dk : %f , cost :%f"%(ct1,ct2,ch1,ch2,di,dj,dk,c))

    return c

X0_global_=np.zeros(7)

sol_global_=minimize(cost_global_,X0_global_,method="SLSQP")
ct1_global,ct2_global,ch1_global,ch2_global,di_global,dj_global,dk_global=sol_global_['x']


Y=compute_acc_global(ct1_global,ct2_global,ch1_global,ch2_global,di_global,dj_global,dk_global)
# %%% Comparison a i j k

f=plt.figure()
ax=f.add_subplot(1,3,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[0]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,0],color="darkred",label="global",alpha=0.5)
ax.legend(),ax.grid()

ax=f.add_subplot(1,3,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[1]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,1],color="darkred",label="global",alpha=0.5)
ax.legend(),ax.grid()

ax=f.add_subplot(1,3,3)
ax.plot(prep_data["t"],prep_data['acc_body_grad[2]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,2],color="darkred",label="global",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
c_i_=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2,axis=0)
c_j_=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2,axis=0)  
c_k_=np.mean((Y[:,2]-prep_data['acc_body_grad[2]'])**2,axis=0)  


s="%f for i \n %f for j \n %f for k"%(c_i_,c_j_,c_k_)
f.suptitle(s)
print(s)


# %% Global 

def compute_eta(vak,omega,c1=c1sol,c2=c2sol):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)
    return eta

def compute_H(vak,omega,ch1,ch2):
    eta=compute_eta(vak,omega)
    H=rho0*Area*(ch1*r0*omega-ch2*(eta-vak))
    return H


def compute_single_motor_thrust_MT(c1,vak,omega,c2=0,vanilla_test=False):
    
    eta=vak/2-r0*omega*c2/4
    eta=eta+0.5*np.sqrt((vak+0.5*r0*omega*c2)**2+2*c1*r0**2*omega**2)

    T=2*rho0*Area*eta*(eta-vak)

    if vanilla_test:
        T=c1*omega**2
    return T

def compute_acc_k(c1,c2=0,df=prep_data,vanilla=False,model="MT"):
    
    vak=df["speed_body[2]"]
    gamma=df["gamma[2]"]
    
    if model=="MT":
        T_sum=sum([compute_single_motor_thrust_MT(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    elif model=="BET":
        T_sum=sum([compute_single_motor_thrust_BET(c1,vak,df['omega_c[%i]'%(i+1)],c2,vanilla_test=vanilla) for i in range(6)])
    else:
        return print("FIX MODEL")
    acc_k=-T_sum/mass+gamma
    
    return acc_k


def compute_acc_global(ct1,ct2,ch1,ch2,di=0,dj=0,dk=0,df=prep_data,vwi=0,vwj=0):
    
    vai=df["speed_body[0]"]
    vaj=df["speed_body[1]"]
    vak=df["speed_body[2]"]
    
    gammai=df["gamma[0]"]
    gammaj=df["gamma[1]"]
    gammak=df["gamma[2]"]
    
    T=sum([compute_single_motor_thrust_MT(ct1,vak,df['omega_c[%i]'%(i+1)],ct2) for i in range(6)])
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    
    H_vect=np.c_[-vai*H,-vaj*H,np.zeros(H.shape)]
    T_vect=np.c_[np.zeros(T.shape),np.zeros(T.shape),T]
    absva=np.sqrt(vai**2+vaj**2+vak**2)
    Fa=-rho0*Area*np.c_[di*absva*vai,dj*absva*vaj,dk*absva*vak]

    return -T_vect/mass+H_vect+np.c_[gammai,gammaj,gammak]+Fa


def cost_global_dij_(X):
    ct1,ct2,ch1,ch2,dij,dk=X

    Y=compute_acc_global(ct1,ct2,ch1,ch2,dij,dij,dk)

    ci=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2/max(abs(prep_data['acc_body_grad[0]']))**2,axis=0)
    cj=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2/max(abs(prep_data['acc_body_grad[1]']))**2,axis=0)
    ck=np.mean((Y[:,2]-prep_data['acc_body_grad[2]'])**2/max(abs(prep_data['acc_body_grad[2]']))**2,axis=0)
    
    c=ci+cj+ck

    print("ct1 :%f, ct2 :%f , ch1 :%f , ch2 :%f , di :%f , dj : %f , dk : %f , cost :%f"%(ct1,ct2,ch1,ch2,dij,dij,dk,c))

    return c

X0_global_dij_=np.zeros(6)

sol_global_dij_=minimize(cost_global_dij_,X0_global_dij_,method="SLSQP")
ct1_global,ct2_global,ch1_global,ch2_global,di_global,dk_global=sol_global_dij_['x']
dj_global=di_global

Y=compute_acc_global(ct1_global,ct2_global,ch1_global,ch2_global,di_global,dj_global,dk_global)
# %%% Comparison a i j k ij equal

f=plt.figure()
ax=f.add_subplot(3,1,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[0]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,0],color="darkred",label="global",alpha=0.5)
ax.legend(),ax.grid()

ax=f.add_subplot(3,1,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[1]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,1],color="darkred",label="global",alpha=0.5)
ax.legend(),ax.grid()

ax=f.add_subplot(3,1,3)
ax.plot(prep_data["t"],prep_data['acc_body_grad[2]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,2],color="darkred",label="global",alpha=0.5)
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
c_i_=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2,axis=0)
c_j_=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2,axis=0)  
c_k_=np.mean((Y[:,2]-prep_data['acc_body_grad[2]'])**2,axis=0)  


s="IJ EQUAL \n %f for i \n %f for j \n %f for k"%(c_i_,c_j_,c_k_)
f.suptitle(s)
print(s)





# %% WITH WIND 

import transforms3d as tf3d


def compute_acc_global_wind(ct1,ct2,ch1,ch2,di=0,dj=0,dk=0,df=prep_data,vwi=0,vwj=0):

    q0,q1,q2,q3=(prep_data['q[0]'],prep_data['q[1]'],
                 prep_data['q[2]'],prep_data['q[3]'])
    
    "precomputing transposition"
    R_transpose=np.array([tf3d.quaternions.quat2mat([i,j,k,l]).T for i,j,k,l in zip(q0,q1,q2,q3)])
    
    vw_earth=np.array([vwi,vwj,0])
    vw_body=R_transpose@vw_earth
    
    
    vai=df["speed_body[0]"]-vw_body[:,0]
    vaj=df["speed_body[1]"]-vw_body[:,1]
    vak=df["speed_body[2]"]-vw_body[:,2]
    
    gammai=df["gamma[0]"]
    gammaj=df["gamma[1]"]
    gammak=df["gamma[2]"]
    
    T=sum([compute_single_motor_thrust_MT(ct1,vak,df['omega_c[%i]'%(i+1)],ct2) for i in range(6)])
    H=sum([compute_H(vak,df['omega_c[%i]'%(i+1)],ch1,ch2) for i in range(6)])
    
    H_vect=np.c_[-vai*H,-vaj*H,np.zeros(H.shape)]
    T_vect=np.c_[np.zeros(T.shape),np.zeros(T.shape),T]
    absva=np.sqrt(vai**2+vaj**2+vak**2)
    Fa=-rho0*Area*np.c_[di*absva*vai,dj*absva*vaj,dk*absva*vak]

    return -T_vect/mass+H_vect+np.c_[gammai,gammaj,gammak]+Fa


def cost_global_dij_wind_(X):
    ct1,ct2,ch1,ch2,dij,dk,vwi,vwj=X

    Y=compute_acc_global_wind(ct1,ct2,ch1,ch2,dij,dij,dk,vwi=vwi,vwj=vwj)

    ci=np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2/max(abs(prep_data['acc_body_grad[0]']))**2,axis=0)
    cj=np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2/max(abs(prep_data['acc_body_grad[1]']))**2,axis=0)
    ck=np.mean((Y[:,2]-prep_data['acc_body_grad[2]'])**2/max(abs(prep_data['acc_body_grad[2]']))**2,axis=0)
    
    c=ci+cj+ck

    print("ct1 :%f, ct2 :%f , ch1 :%f , ch2 :%f , di :%f , dj : %f , dk : %f , vwi : %f ,vwj : %f cost :%f"%(ct1,ct2,ch1,ch2,dij,dij,dk,vwi,vwj,c))

    return c


X0_global_dij_wind_=np.zeros(8)

sol_global_dij_wind_=minimize(cost_global_dij_wind_,X0_global_dij_wind_,method="SLSQP")
ct1_global,ct2_global,ch1_global,ch2_global,di_global,dk_global,vwi_global_,vwj_global_=sol_global_dij_wind_['x']
dj_global=di_global

Y=compute_acc_global_wind(ct1_global,ct2_global,ch1_global,ch2_global,di_global,dj_global,dk_global,vwi=vwi_global_,vwj=vwj_global_)



f=plt.figure()
ax=f.add_subplot(3,1,1)
ax.plot(prep_data["t"],prep_data['acc_body_grad[0]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,0],color="darkred",label="global")
ax.legend(),ax.grid()

ax=f.add_subplot(3,1,2)
ax.plot(prep_data["t"],prep_data['acc_body_grad[1]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,1],color="darkred",label="global")
ax.legend(),ax.grid()

ax=f.add_subplot(3,1,3)
ax.plot(prep_data["t"],prep_data['acc_body_grad[2]'],color="black",label="log")
ax.plot(prep_data["t"],Y[:,2],color="darkred",label="global")
ax.legend(),ax.grid()

print("\nPerformances: ")
print("RMS error on acc pred is : ")
# c_i_=np.sqrt(np.mean((Y[:,0]-prep_data['acc_body_grad[0]'])**2,axis=0))
# c_j_=np.sqrt(np.mean((Y[:,1]-prep_data['acc_body_grad[1]'])**2,axis=0)  )
# c_k_=np.sqrt(np.mean((Y[:,2]-prep_data['acc_body_grad[2]'])**2,axis=0)  )

c_i_=np.mean(np.abs(Y[:,0]-prep_data['acc_body_grad[0]']),axis=0)
c_j_=np.mean(np.abs(Y[:,1]-prep_data['acc_body_grad[1]']),axis=0)  
c_k_=np.mean(np.abs(Y[:,2]-prep_data['acc_body_grad[2]']),axis=0)  
s="WIND \n %f for i \n %f for j \n %f for k"%(c_i_,c_j_,c_k_)
f.suptitle(s)
print(s)






# %% Synthesis


bilan=pd.DataFrame(data=None,
                    columns=['ct1','ct2',
                            'ch1','ch2',
                            'di','dj','dk','vwi','vwj',
                            'cost'],
                    index=['vanilla','custom',
                          'vanilla_dk','custom_with_dk',
                          'ai_drag','ai_h','ai_drag_and_h',
                          'aj_drag','aj_h','aj_drag_and_h',
                          'aij_h','aij_h_and_drag',
                          'aij_h_drag_equal_coeffs',
                          "global","global_equal_coeffs","global_wind"])




bilan.loc["vanilla"]['ct1','cost']=np.r_[sol_vanilla['x'],sol_vanilla['fun']]
bilan.loc["custom"]['ct1','ct2','cost']=np.r_[sol_custom['x'],sol_custom['fun']]

bilan.loc["vanilla_dk"]['ct1','dk','cost']=np.r_[sol_vanilla_drag['x'],sol_vanilla_drag['fun']]
bilan.loc["custom_with_dk"]['ct1','ct2','dk','cost']=np.r_[sol_custom_drag['x'],sol_custom_drag['fun']]

bilan.loc['ai_drag']['di','cost']=np.r_[sol_ai_od['x'],sol_ai_od['fun']]
bilan.loc['ai_h']['ch1','ch2','cost']=np.r_[sol_ai_oh['x'],sol_ai_oh['fun']]
bilan.loc['ai_drag_and_h']['ch1','ch2','di','cost']=np.r_[sol_ai_hwd['x'],sol_ai_hwd['fun']]

bilan.loc['aj_drag']['dj','cost']=np.r_[sol_aj_od['x'],sol_aj_od['fun']]
bilan.loc['aj_h']['ch1','ch2','cost']=np.r_[sol_aj_oh['x'],sol_aj_oh['fun']]
bilan.loc['aj_drag_and_h']['ch1','ch2','dj','cost']=np.r_[sol_aj_hwd['x'],sol_aj_hwd['fun']]


bilan.loc['aij_h']['ch1','ch2','cost']=np.r_[sol_aij_nodrag['x'],sol_aij_nodrag['fun']]
bilan.loc['aij_h_and_drag']['ch1','ch2','di','dj','cost']=np.r_[sol_aij_hwd['x'],sol_aij_hwd['fun']]

bilan.loc['aij_h_drag_equal_coeffs']['ch1','ch2','di','cost']=np.r_[sol_aij_hwd_di_eq_dj_['x'],sol_aij_hwd_di_eq_dj_['fun']]
bilan.loc['aij_h_drag_equal_coeffs']['dj']=bilan.loc['aij_h_drag_equal_coeffs']['di']

bilan.loc['global']['ct1','ct2',
                    'ch1','ch2',
                    'di','dj','dk',
                    'cost']=np.r_[sol_global_['x'],sol_global_['fun']]

bilan.loc['global_equal_coeffs']['ct1','ct2',
                    'ch1','ch2',
                    'di','dk',
                    'cost']=np.r_[sol_global_dij_['x'],sol_global_dij_['fun']]

bilan.loc['global_equal_coeffs']["dj"]=bilan.loc['global_equal_coeffs']["di"]

bilan.loc['global_wind']['ct1','ct2',
                    'ch1','ch2',
                    'di','dk','vwi','vwj',
                    'cost']=np.r_[sol_global_dij_wind_['x'],sol_global_dij_wind_['fun']]
bilan.loc['global_wind']["dj"]=bilan.loc['global_wind']["di"]




print(bilan)


