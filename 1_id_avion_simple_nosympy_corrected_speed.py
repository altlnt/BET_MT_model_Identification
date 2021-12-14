#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:59:50 2021

@author: alex
"""
import numpy as np
import transforms3d as tf3d
import scipy
from scipy import optimize
from sklearn.utils import gen_batches
import random


# %%   ####### IMPORT DATA 

import pandas as pd


log_path="./logs/avion/vol123/log_real_processed.csv"


    
raw_data=pd.read_csv(log_path)



prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
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

data_prepared.insert(data_prepared.shape[1],'omega_c[5]',(data_prepared['PWM_motor[5]']-1000)*925.0/1000)
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
upwards.append(Rvd@np.array([0.0,0,-1.0]))
upwards.append(Rvg@np.array([0.0,0,-1.0]))

crosswards=[np.cross(i,j) for i,j in zip(forwards,upwards)]

alpha_0=0.07
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
C_t0 = 1.1e-4
C_q = 1e-8
C_h = 1e-4

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
    df.insert(df.shape[1],
              'speed_body[%i]'%(i),
              v_body_array[:,i])
    df.insert(df.shape[1],
              'gamma[%i]'%(i),
              gamma_array[:,i])
    df.insert(df.shape[1],
              'omega[%i]'%(i),
              omegas_new[:,i])
    
df.insert(df.shape[1],
          'thrust_dir_ned',
          [i[:,0]*j**2 for i,j in zip(df['R'],df['omega_c[5]'])])

import numpy as np
delt=np.array([df['PWM_motor[%i]'%(i)] for i in range(1,5)]).T
delt=np.concatenate((np.zeros((len(df),1)),delt),axis=1).reshape(-1,1,5)
delt=(delt-1530)/500*15.0/180.0*np.pi
delt[:,:,0]*=0
delt[:,:,2]*=-1.0
delt[:,:,4]*=-1.0

df.insert(df.shape[1],
          'deltas',
          [i for i in delt])

# %% usefuncs

alpha_0=0.07
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
C_t0 = 1.1e-4
C_q = 1e-8
C_h = 1e-4

ct = 1.1e-4
a_0 =  0.07
a_s =  0.3391
d_s =  15.0*np.pi/180
cl1sa = 5
cd1fp = 2.5
k0 = 0.1
k1 = 0.1
k2 = 0.1
cd0fp =  1e-2
cd0sa = 0.3
cd1sa = 1.0
m= 8.5

coeffs_0=np.array([ct,
                   a_0,
                   a_s,
                   d_s, 
                   cl1sa, 
                   cd1fp, 
                   k0, k1, k2, 
                   cd0fp, 
                   cd0sa, cd1sa,m])

# %% OPTI SIMPLE
# %%% usefuncs
def pred(df_arg=df,coeffs=coeffs_0,fix_mass=False,fix_ct=False):
    
    ct,a_0, a_s, d_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, cd0sa, cd1sa,m=coeffs

    ct= 2.0*1.1e-4 if fix_ct else ct
    m= 8.5 if fix_mass else m
    

    v_pred_array=np.zeros((len(df_arg),3))
    v_pred_current=np.array([df_arg['speed[%i]'%(i)] for i in range(3)]).T[0]
    R_array=np.array([i for i in df_arg["R"]])
    omegas=np.array([ df_arg['omega[%i]'%(i)] for i in range(3)]).T
    
    for i,k in enumerate(df_arg.index):

        
        d_0=np.array([l for l in df_arg['deltas']])[i].reshape((-1,))

        v_in_ldp=np.cross(crosswards,np.cross((v_pred_current-np.cross(cp_list,omegas[i])),crosswards))
        
        dd=-v_in_ldp
        dd=dd.T@np.diag(1.0/(np.linalg.norm(dd,axis=1)+1e-4))
    
        ld=np.cross(crosswards,v_in_ldp)
        ld=ld.T@np.diag(1.0/(np.linalg.norm(ld,axis=1)+1e-4))
                  
        sd=-(v_pred_current-np.cross(cp_list,omegas[i])-v_in_ldp)
        sd=sd.T@np.diag(1.0/(np.linalg.norm(sd,axis=1)+1e-4))
        
        dragdirs=R_array[i]@(dd@np.diag(Aire_list)*np.linalg.norm(v_in_ldp)**2)
        liftdirs=R_array[i]@(ld@np.diag(Aire_list)*np.linalg.norm(v_in_ldp)**2)

        
        alphas_d=np.diag(v_in_ldp@(np.array(forwards).T))/(np.linalg.norm(v_in_ldp,axis=1)+1e-4)
        alphas_d=np.arccos(alphas_d)
        alphas_d=np.sign(np.diag(v_in_ldp@np.array(upwards).T))*alphas_d
        
        a=alphas_d
    
    
        CL_sa = 1/2 * cl1sa * np.sin(2*(a + (k1*d_0) + a_0))
        CD_sa = cd0sa + cd1sa * np.sin((a + (k0*d_0) + a_0))**2
    
        CL_fp = 1/2 * cd1fp * np.sin((2*(a+ (k1*d_0) + a_0)))
        CD_fp = cd0fp + cd1fp * np.sin((a + (k0*d_0) + a_0))**2
    
        puiss=5
        s = 1.0 - ((a+a_0)**2/a_s**2)**puiss/(((a+a_0)**2/a_s**2)**puiss + 100+200*d_s)
    
        C_L = CL_fp + s*(CL_sa - CL_fp) + k2 * np.sin(d_0)
        C_D = CD_fp + s*(CD_sa - CD_fp)
        
        #C_L,C_D shape is (n_samples,1,n_surfaces)
        
        lifts=C_L*liftdirs    
        drags=C_D*dragdirs
        
        aeroforce_total=np.sum(lifts+drags)
        
        # "compute thrust  "
    
        T=ct*np.array([l for l in df_arg['thrust_dir_ned']])[i]

        g=np.array([0,0,9.81])
        forces_total=T+aeroforce_total+m*g
        acc=forces_total/m
        new_v_pred=v_pred_current+min(1e-2,df_arg['dt'][k])*acc
        new_v_pred=np.nan_to_num(new_v_pred)
        v_pred_array[i]=new_v_pred
        v_pred_current=R_array[i].T@new_v_pred
    
    return v_pred_array

counter=0
def cost(X,fm=False,fct=False,nsamples=1000):

    global counter
    
    X0=X*coeffs_0
    
    indexes_batches=list(gen_batches(len(df),len(df)//nsamples))
    
    idb=indexes_batches[:-1]
    random.shuffle(idb)
    
    df_batches=[df[i] for i in idb]
    c=0
    for j,df_temp in enumerate(df_batches):
        vpred=pred(df_arg=df_temp,coeffs=X0,fix_mass=False,fix_ct=False)
        vlog=np.array([df_temp['speed[%i]'%(i)] for i in range(3)]).T
        c+=np.mean(np.linalg.norm((vpred-vlog),axis=1))
        counter+=1
        print("cost : %f ,counter %i : %i / %i "%(c,counter,j,len(df_batches)))
        
        
    list_to_print=[i for i in X]+c
    print(str(list_to_print))
    str_top_print="\r "
    for i in X:
        str_top_print=str_top_print+str(round(i,ndigits=5))+" |"
    str_top_print=str_top_print+" "+str(round(c,ndigits=5))
    
    res={}
    l="ct,a_0, a_s, d_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, cd0sa, cd1sa,m"
    for i,j in zip(l.split(","),X0):
        res[i]=round(j,ndigits=5)       
    res['cost']=c
    print(res)
    return c

scipy_init_x=coeffs_0


# sol=scipy.optimize.minimize(cost,scipy_init_x,args=("True","False"))

# sol=cscipy.optimize.minimize(cost,scipy_init_x,args=("False","True"))

sol=scipy.optimize.minimize(cost,scipy_init_x,args=("True","True"))

# sol=scipy.optimize.minimize(cost,scipy_init_x,method="SLSQP")

# sol=scipy.optimize.minimize(cost,scipy_init_x)

