
import numpy as np 
import os 
from numba import jit
 
fit_arg,blr,ns=False,0.1,'all'

log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
# les booleans qui déterminent plusieurs comportements, voir coms     

fit_on_v=fit_arg #est ce qu'on fit sur la vitesse prédite ou sur l'acc
used_logged_v_in_model=not fit_arg
bo=False

# si on utilise l'optimizer scipy, on passe 'scipy' en argument
# sinon le learning rate de la descente de gradient est base_lr

base_lr=1.0 if  blr=="scipy" else blr
fit_strategy="scipy" if  blr=="scipy" else "custom_gradient"

# si wind_signal, on considère la valeur du vent à chaque instant
# est un paramètre d'optimisation
# sinon, le vent est considéré comme une constante

wind_signal=False
assume_nul_wind=False # if true, le signal de vent constant vaut zéro
nsecs=ns
# nsecs désigne la taille du batch en secondes



# n_epochs=20 #le nombre d'epochs

# log_name désigne le nom des fichiers que l'on enregistrera
# on customisera chaque fichier avec le numero de l'epoch et la timestamp
# log_name="3_SEPTEMBRE_fit_v_%s_lr_%s_ns_%s"%(str(fit_on_v),str(base_lr) if blr!="scipy" else 'scipy',str(ns))

#                   CI DESSOUS LES PARAMETRES PROPRES AU MODELE



#                  Ci dessous, on décide quels paramètres on souhaite identifier
# id_mass=False
# id_wind=not assume_nul_wind



train_proportion=1.0 #proportion data train vs validation

# log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
# save_dir_name="results"

# Paramètres utilitaires

mass=8.5
g=np.array([0,0,9.81])

Aire_1,Aire_2,Aire_3,Aire_4,Aire_5 =    0.62*0.262* 1.292 * 0.5,\
                                    0.62*0.262* 1.292 * 0.5, \
                                    0.34*0.01* 1.292 * 0.5,\
                                    0.34*0.1* 1.292 * 0.5, \
                                    1.08*0.31* 1.292 * 0.5
Aire_list = [Aire_1,Aire_2,Aire_3,Aire_4,Aire_5]
cp_1,cp_2,cp_3,cp_4,cp_5 = np.array([-0.013,0.475,-0.040],       dtype=float).flatten(), \
                        np.array([-0.013,-0.475,-0.040],      dtype=float).flatten(), \
                        np.array([-1.006,0.85,-0.134],    dtype=float).flatten(),\
                        np.array([-1.006,-0.85,-0.134],   dtype=float).flatten(),\
                        np.array([0.021,0,-0.064],          dtype=float).flatten()
cp_list=[cp_1,cp_2,cp_3,cp_4,cp_5]

Area=np.pi*(11.0e-02)**2
r0=11e-02
rho0=1.204
kv_motor=800.0
pwmmin=1075.0
pwmmax=1950.0
U_batt=16.8

vwi0=0.0
vwj0=0.0
vwk0=0.0

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
C_t = 1.1e-4
C_q = 1e-8
C_h = 1e-4


# alpha_0= 0* 0.07
# alpha_s =  0.3391428111
# delta_s = 15.0*np.pi/180
# cd0sa_0 = 0* 0.010
# cd0fp_0 = 0* 0.010
# cd1sa_0 = 0* 4.55 
# cl1sa_0 = 0* 5 
# cd1fp_0 = 0* 2.5 
# coeff_drag_shift_0= 0*0.5 
# coeff_lift_shift_0= 0*0.05 
# coeff_lift_gain_0= 0*2.5
# C_t = 1.1e-4
# C_q = 0*1e-8
# C_h = 0*1e-4





physical_params=[mass,
Area,
r0,
rho0,
kv_motor,
pwmmin,
pwmmax,
U_batt,
cd0sa_0,
cd0fp_0,
cd1sa_0,
cl1sa_0 ,
cd1fp_0,
coeff_drag_shift_0,
coeff_lift_shift_0,
coeff_lift_gain_0,
vwi0,
vwj0]


# Bounds and scaling factors
bounds={}
bounds['m']=(0,np.inf)
bounds['A']=(0,np.inf)
bounds['r']=(0,np.inf)
bounds['cd0sa']=(0,np.inf)
bounds['cd0fp']=(-np.inf,np.inf)
bounds['cd1sa']=(-np.inf,np.inf)
bounds['cl1sa']=(-np.inf,np.inf)
bounds['cd1fp']=(-np.inf,np.inf)
bounds['coeff_drag_shift']=(0,np.inf)
bounds['coeff_lift_shift']=(0,np.inf)
bounds['coeff_lift_gain']=(0,np.inf)
bounds['vw_i']=(-15,15)
bounds['vw_j']=(-15,15)

"scaler corresponds roughly to the power of ten of the parameter"
"it does not have to though, it may be used to improve the grad descent"

# scalers={}
# for i in bounds:
# scalers[i]=1.0







# %%   ####### Saving function

# import json
# import datetime

# #generating a new log name if none is provided
# if log_name=="":
# # log_name=str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

# # spath=os.path.join(os.getcwd(),save_dir_name,log_name)

# #on enlève les logs avec le même nom
# import shutil
# try:
# shutil.rmtree(spath)
# except:
# pass
# os.makedirs(spath)

# #on sauvegarde les paramètres de départ
# with open(os.path.join(spath,'data.json'), 'w') as fp:
# json.dump(metap, fp)

# import pandas as pd

# # la fonction saver va être utilisée souvent 

# def saver(name=None,save_path=os.path.join(os.getcwd(),"results_tests"),**kwargs):
# dname=str(int(time.time())) if (name is None) else name

# "le fonctionnement est simple:on prend tout ce qu'il y a dans **kwargs"
# "et  on le met dans un dictionnaire qu'on save dans un .json"

# D={}
# for i in kwargs:
#     if type(kwargs[i])==dict:
#         for j in kwargs[i]:
#             D[j]=kwargs[i][j]
#     else:
#         D[i]=kwargs[i].tolist() if isinstance(kwargs[i],np.ndarray) else kwargs[i]

# with open(os.path.join(save_path,'%s.json'%(dname)), 'w') as fp:
#     json.dump(D, fp)
# return 0 


# %%   ####### SYMPY PROBLEM 

" le but de cette partie est de génerer une fonction qui renvoie : "
" new_acc,new_v,sqerr_a,sqerr_v,Ja.T,Jv.T    , c'est à dire        "
" l'acc pred, la vitesse pred, l'erreur quad sur acc, l'erreur quad sur v"
" la jacobienne de l'erreur quad sur l'acc et la vitesse "

@jit(nogil=True)
def Rotation(R,angle):
    c, s = np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)
    r = np.array([[1,0, 0], [0,c, -s],[0,s, c]] , dtype=np.float)
    return R @ r
#CI DESSOUS : on spécifie quelles variables sont les variables d'identif
   
import dill
model_func = dill.load(open('./.Funcs/model_func_'+str(used_logged_v_in_model)+"simple_"+str(bo),'rb'))[0]
function_moteur_physique=  dill.load(open('./.Funcs/model_func_'+str(used_logged_v_in_model)+"simple_"+str(bo),'rb'))[1]
# CI DESSOUS : on spécifie quelles variables sont les variables d'identif
import time
t7=time.time()

"cleansing memory"

# très recommandé d'effacer les variables de sympy pour éviter les soucis 
# dans le restes des fonctions

# del(dt,m,cost_scaler_a,cost_scaler_v,
# vlog_i,vlog_j,vlog_k,
# vpred_i,vpred_j,vpred_k,
# alog_i,alog_j,alog_k,
# vnext_i,vnext_j,vnext_k,
# vw_i,vw_j,
# kt,
# b1,
# c1,c2,c3,
# ch1,ch2,
# di,dj,dk,
# rho,A,r,r1,r2,r3,r4,r5,r6,r7,r8,r9,R,
# omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
# omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6)


"test "



# %%   ####### IMPORT DATA 

# a la fin de ce bloc, on obtient une liste de datasets, correspondant
# aux batches. 

print("LOADING DATA...")
import pandas as pd

raw_data=pd.read_csv(log_path)

print("PROCESSING DATA...")


prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])


# on choisit tmin et tmax selon quel log on utilise

# if "vol12" in log_path:
#     tmin,tmax=(-1,1e10) 
# elif "vol1" in log_path:
#     tmin,tmax=(41,265) 
# elif "vol2" in log_path:
#     tmin,tmax=(10,140) 
# tmin=1
# tmax=500

# prep_data=prep_data[prep_data['t']>tmin]
# prep_data=prep_data[prep_data['t']<tmax]
prep_data=prep_data.reset_index()


for i in range(3):
    prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]


prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
prep_data['t']-=prep_data['t'][0]
prep_data=prep_data.drop(index=[0,len(prep_data)-1])
prep_data=prep_data.reset_index()

data_prepared=prep_data[:len(prep_data)//50]
data_prepared=prep_data


def scale_to_01(df):
    
    return (df-df.min())/(df.max()-df.min())

for i in range(6):
    data_prepared['omega_c[%i]'%(i+1)]=scale_to_01(data_prepared['PWM_motor[%i]'%(i+1)])*925

"splitting the dataset into nsec"
data_batches=[data_prepared]
N_train_batches=1
N_val_batches=0

    
print("DATA PROCESS DONE")


# print("Importing model func...")
# %%   ####### Identification Data Struct

# On répartit les variables entre deux dicts: id_variables et non_id_variables
# Cette étape est liée, par les booléens utilisés, à la premi_re étape

non_id_variables={"m":mass,
              "A":Area,
              "r":r0,
              "rho":rho0,
            'cd0sa':cd0sa_0,
            'cd0fp':cd0fp_0,
            'cd1sa':cd1sa_0,
            'cl1sa':cl1sa_0,
            'cd1fp':cd1fp_0,
            'coeff_drag_shift':coeff_drag_shift_0,
            'coeff_lift_shift':coeff_lift_shift_0,
            'coeff_lift_gain':coeff_lift_gain_0,
              "vw_i":vwi0,
              "vw_j":vwj0,
              "vw_k":vwk0,
              "alpha_stall":alpha_s,
              "largeur_stall":delta_s,
              "Ct": C_t, 
              "Cq": C_q, 
              "Ch": C_h
              }
id_variables={}
for key_ in ('cd0sa','cd0fp',
         'cd1sa','cl1sa','cd1fp',
         'coeff_drag_shift','coeff_lift_shift',
         'coeff_lift_gain'):
    id_variables[key_]=non_id_variables[key_]





if wind_signal:
    id_variables['vw_i']=vwi0*np.zeros(len(data_prepared))
    id_variables['vw_j']=vwj0*np.zeros(len(data_prepared)) 
    

"cleaning non_id_variables to avoid having variables in both dicts"
rem=[]
for i in non_id_variables.keys():
    if i in id_variables.keys():
        rem.append(i)
    
for j in rem:
    del(non_id_variables[j])

# for i in id_variables.keys():
#     id_variables[i]=id_variables[i]/scalers[i]


# %%   ####### MODEL function

# ici, on définit les fonctions que l'on appellera dans la partie 
# optimisation. 


import transforms3d as tf3d 
# import copy 
@jit(nogil=True)
def arg_wrapping(batch,id_variables,data_index,speed_pred_previous):
    
    "cette fonction sert à fabriquer, à partir des inputs, l'argument que "
    "l'on enverra en input à la fonction lambdifiée de la partie sympy    "
    
    "batch est un dataframe"
    "id_variables sont les variables d'identification"
    "scalers sont les coefficients de mise à l'échelle"
    "data_index est un entier, qui sert à récupérer la bonne valeur "
    "dans le batch de données"
    "speed_pred_previous est la vitesse prédite précédente"
    "omegas_pred est la vitesse angulaire pred précédente,à partir de kt"
    
    i=data_index
    
    # cost_scaler_v=1.0
    # cost_scaler_a=1.0
    
    dt=min(batch['dt'][i],1e-2)
    
    vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
    v_log = np.array([[vlog_i],
                   [vlog_j],
                   [vlog_k]])
    
    vpred_i,vpred_j,vpred_k=speed_pred_previous 
    v_pred=np.array([[vpred_i],
                   [vpred_j],
                   [vpred_k]])
    
    alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]
    alog=np.array([[alog_i],
                   [alog_j],
                   [alog_k]])
    
    # vnext_i,vnext_j,vnext_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
    
    m=non_id_variables['m'] if 'm' in non_id_variables else id_variables['m']
    
    vw_i=non_id_variables['vw_i'] if 'vw_i' in non_id_variables else id_variables['vw_i']
    vw_j=non_id_variables['vw_j'] if 'vw_j' in non_id_variables else id_variables['vw_j']
    
    if wind_signal:
        vw_i,vw_j=vw_i[i],vw_j[i]
        
    v_W=np.array([[vw_i],
                 [vw_j], 
                 [vwk0]])
    
    Omega=np.zeros(3)
    
    cd0sa=non_id_variables['cd0sa'] if 'cd0sa' in non_id_variables else id_variables['cd0sa']
    cd0fp=non_id_variables['cd0fp'] if 'cd0fp' in non_id_variables else id_variables['cd0fp']
    cd1sa=non_id_variables['cd1sa'] if 'cd1sa' in non_id_variables else id_variables['cd1sa']
    cl1sa=non_id_variables['cl1sa'] if 'cl1sa' in non_id_variables else id_variables['cl1sa']
    cd1fp=non_id_variables['cd1fp'] if 'cd1fp' in non_id_variables else id_variables['cd1fp']
    coeff_drag_shift=non_id_variables['coeff_drag_shift'] if 'coeff_drag_shift' in non_id_variables else id_variables['coeff_drag_shift']
    coeff_lift_shift=non_id_variables['coeff_lift_shift'] if 'coeff_lift_shift' in non_id_variables else id_variables['coeff_lift_shift']
    coeff_lift_gain=non_id_variables['coeff_lift_gain'] if 'coeff_lift_gain' in non_id_variables else id_variables['coeff_lift_gain']
    
    R=tf3d.quaternions.quat2mat(np.array([batch['q[%i]'%(j)][i] for j in range(4)]))
    R_list =[R,R,Rotation(R,-45),Rotation(R,-135),R]
    
    
    "reverse mixing"
    pwm_null_angle=1527
    RW_delta=batch['PWM_motor[1]'][i]-pwm_null_angle
    LW_delta=batch['PWM_motor[2]'][i]-pwm_null_angle
    RVT_delta=batch['PWM_motor[3]'][i]-pwm_null_angle
    LVT_delta=batch['PWM_motor[4]'][i]-pwm_null_angle
    
    delta0_list=np.array([RW_delta,-LW_delta,RVT_delta,-LVT_delta,0.0])  
    
    delta_pwm=500
    delta0_list/=delta_pwm
    
    delta0_list=delta0_list*15.0/180.0*np.pi
    
    ## Commande entre -15:15 pour les 4 premier terme, le dernier terme vaut 0 (pour l'homogéinité des longueur)
    omega_rotor = batch['omega_c[5]'][i]                    ## Vitesse de rotation des helices (supposé les mêmes pour les deux moteurs)
    alpha_list=[0,0,0,0,0]
    for p, cp in enumerate(cp_list) :          # Cette boucle calcul les coefs aéro pour chaque surface 
        VelinLDPlane   = function_moteur_physique[0](Omega, cp, v_pred.flatten(), v_W.flatten(), R_list[p].flatten())
        dragDirection  = function_moteur_physique[1](Omega, cp, v_pred.flatten(), v_W.flatten(), R_list[p].flatten())
        liftDirection  = function_moteur_physique[2](Omega, cp, v_pred.flatten(), v_W.flatten(), R_list[p].flatten())
        alpha_list[p] =  -function_moteur_physique[3](dragDirection, liftDirection, np.array([[1],[0],[0]]).flatten(), VelinLDPlane)

    X=(alog.flatten(),v_log.flatten(),dt, Aire_list, Omega.flatten(), R.flatten(), v_pred.flatten(), v_W.flatten(), cp_list, alpha_list, alpha_0, \
       alpha_s, delta0_list.flatten(), delta_s, cl1sa, cd1fp, coeff_drag_shift, coeff_lift_shift, coeff_lift_gain,\
           cd0fp, cd0sa, cd1sa, C_t, C_q, C_h, omega_rotor, g.flatten(), m)
    
    return X

@jit(nogil=True)
def pred_on_batch(batch,id_variables):
    
    "si n est la taille du batch"
    "cette fonction sert à appeler n fois la fonction lambdifiée"
    " de sympy "
    
    "on obtient n acc prédites, n vitesses prédites, n jacobiennes...."
    
    
    "batch est un dataframe"
    "id_variables sont les variables d'identification"
    "scalers sont les coefficients de mise à l'échelle"
    
    acc_pred=np.zeros((len(batch),3))
    speed_pred=np.zeros((len(batch),3))
    
        
    square_error_a=np.zeros((len(batch),1))    
    square_error_v=np.zeros((len(batch),1))    
    jac_error_a=np.zeros((len(batch),len(id_variables)))
    jac_error_v=np.zeros((len(batch),len(id_variables)))
    
    for i in batch.index:
    
        print("\r Pred on batch %i / %i "%(i,max(batch.index)), end='', flush=True)
    
        speed_pred_prev=speed_pred[i-1] if i>min(batch.index) else (batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i])
    
        X=arg_wrapping(batch,id_variables,i,speed_pred_prev)

        Y=model_func(*X)
        # print(Y)
        acc_pred[i]=Y[:3].reshape(3,)
        speed_pred[i]=Y[3:6].reshape(3,)
        square_error_a[i]=Y[6:7].reshape(1,)
        square_error_v[i]=Y[7:8].reshape(1,)
        jac_error_a[i]=Y[8:8+len(id_variables)].reshape(len(id_variables),)
        jac_error_v[i]=Y[8+len(id_variables):8+2*len(id_variables)].reshape(len(id_variables),)
        
    
    
    return acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v



# %%   Gradient
# import random



def X_to_dict(X,base_dict=id_variables):
    
    "sert à transformer un vecteur en dictionnaire "
    
    out_dict={}
    index_j=0
    for i,key in enumerate(base_dict.keys()):
            out_dict[key]=X[index_j:index_j+len(base_dict[key])] if isinstance(base_dict[key],np.ndarray) else X[index_j] 
            index_j+=len(base_dict[key]) if isinstance(base_dict[key],np.ndarray) else 1
    return out_dict

def dict_to_X(input_dict):

    "sert à transformer un dictinonaire en vecteur "
    
    out=np.r_[tuple([np.array(input_dict[key]).flatten() for key in input_dict])]
    return out

def prepare_dict(id_var,index):
    "sert à processer le dictionnaire id_var"
    "si wind_signal, on ne retient que la valeur index de id_var"
    
    if not wind_signal:
        return id_var
    else:
        newdic={}
        for k in id_var:
            newdic[k]=id_var[k]
        
        newdic['vw_i'],newdic['vw_j']=newdic['vw_i'][index],newdic['vw_j'][index]
        return newdic

def propagate_gradient(jac_array,id_variables,
                   lr=base_lr):
    "à partir de n jacobiennes et du dictionnaire des variables "
    "d'identification, on calcule les nouveaux paramètres"
    "cette fonction est la fonction que l'on utilisera si on choisit"
    "la descente de gradient comme algorithme de minmisation"
    
    X=dict_to_X(id_variables)
    
    # print(" --- Propagate %s grad : "%(log_name))


    J=np.zeros(len(X))
    index_j=0
    for i,key in enumerate(id_variables.keys()):
        if type(id_variables[key]) is not np.ndarray:
    
            J[index_j]=np.mean(jac_array[:,i])
            index_j+=1
        else:
            J[index_j:index_j+len(id_variables[key])]=jac_array[:,i]
            index_j+=len(id_variables[key])
    
    new_X=X-lr*J
    new_dic=X_to_dict(new_X,id_variables)
    return new_dic

# %% Train loop


        
acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(data_prepared[:500],id_variables)
total_sc_a=np.sqrt(np.mean(square_error_a,axis=0))
total_sc_v=np.sqrt(np.mean(square_error_v ,axis=0))
    

import matplotlib.pyplot as plt

plt.figure()
for i in range(3):
    
    ax=plt.gcf().add_subplot(4,1,i+1)
    ax.plot(data_prepared['t'],data_prepared['acc[%i]'%(i)],color="black",label="data")
    ax.plot(data_prepared['t'][np.arange(len(acc_pred))],acc_pred[:,i],color="red",label="pred")
    plt.grid()
    
ax=plt.gcf().add_subplot(4,1,4)
ax.plot(data_prepared['t'],data_prepared['omega_c[5]'],color="black",label="data")
# ax.plot(data_prepared['t'],acc_pred[:,i],color="red",label="pred")
plt.grid()


plt.figure()
for i in range(3):
    
    ax=plt.gcf().add_subplot(3,1,i+1)
    ax.plot(data_prepared['t'],data_prepared['speed[%i]'%(i)],color="green",label="data")
    plt.grid()
    
    
    
    
