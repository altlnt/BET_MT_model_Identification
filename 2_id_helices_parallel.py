

import  numpy as np
# %%   ####### PARAMETERS
import sys
import time
from sympy import *
import gc 


"cette fonction est le main"
"on utilise le multiprocessing pour tester plusieurs metaparamètres"
"elle est appelée par le process"


def main_func(x):
    
    # récupération des arguments
    
    ns,with_ct3,vanilla_force_model,assume_nul_wind,di_equal_dj=x

    
    blr='scipy'
    fit_arg=True
    
    ### TEST
    
    # fit_arg=True
    # blr=1
    # ns=2.0
    # with_ct3=False
    # vanilla_force_model=False
    # wind_signal=False
    # assume_nul_wind=True
    
    
    # les booleans qui déterminent plusieurs comportements, voir coms     

    fit_on_v=fit_arg #est ce qu'on fit sur la vitesse prédite ou sur l'acc
    used_logged_v_in_model=not fit_arg
    
    # si on utilise l'optimizer scipy, on passe 'scipy' en argument
    # sinon le learning rate de la descente de gradient est base_lr
    
    base_lr=1.0 if  blr=="scipy" else blr
    fit_strategy="scipy" if  blr=="scipy" else "custom_gradient"

    # si wind_signal, on considère la valeur du vent à chaque instant
    # est un paramètre d'optimisation
    # sinon, le vent est considéré comme une constante
    
    wind_signal=False
    
    # assume_nul_wind=False # if true, le signal de vent constant vaut zéro
    nsecs=ns
    # nsecs désigne la taille du batch en secondes
    
    model_motor_dynamics=False #si true, on fait intervernir kT

    n_epochs=20 #le nombre d'epochs
    
    # log_name désigne le nom des fichiers que l'on enregistrera
    # on customisera chaque fichier avec le numero de l'epoch et la timestamp
    log_name="30_oct_fit_v_%s_lr_%s_ns_%s"%(str(fit_on_v),str(base_lr) if blr!="scipy" else 'scipy',str(ns))
    for i in 'with_ct3,vanilla_force_model,assume_nul_wind,di_equal_dj'.split(','):
        log_name=log_name+"_%s_%s"%(i,str(eval(i)))
    #                   CI DESSOUS LES PARAMETRES PROPRES AU MODELE
    
    # with_ct3=False
    # vanilla_force_model=False
    
    structural_relation_idc1=False
    structural_relation_idc2=False
    
    if vanilla_force_model and (structural_relation_idc1 or structural_relation_idc2):
        print("INVALID :")
        print("vanilla force model and structural_relation cannot be true at the same time")
        sys.exit()
        
    if structural_relation_idc1 and structural_relation_idc2:
        print("INVALID :")
        print("structural_relation_idc1 and structural_relation_idc2 cannot be true at the same time")
        sys.exit()
    
    approx_x_plus_y=False

    #                  Ci dessous, on décide quels paramètres on souhaite identifier
    id_mass=False
    id_blade_coeffs=True
    id_c3=with_ct3
    id_blade_geom_coeffs=False
    id_body_liftdrag_coeffs=True
    id_wind=not assume_nul_wind
    id_time_const=model_motor_dynamics
    
    
    
    train_proportion=0.8 #proportion data train vs validation
    
    log_path="./logs/copter/vol12/log_real_processed.csv"
    save_dir_name="res_helices_parallel_01_nov/bounds"

    # Paramètres utilitaires
    
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
    
    c10=0.0139055
    c20=-0.0386786
    c30=0.0
    ch10=0.0282942
    ch20=-0.1
    di0=0.806527
    dj0=0.632052
    dk0=1.59086
    
    vwi0=0.0
    vwj0=0.0
    
    kt0=5.0
    
    physical_params=[mass,
    Area,
    r0,
    rho0,
    kv_motor,
    pwmmin,
    pwmmax,
    U_batt,
    b10,
    c10,
    c20,
    c30,
    ch10,
    ch20,
    di0,
    dj0,
    dk0,
    vwi0,
    vwj0,
    kt0]
    
    
    # Bounds and scaling factors
    bounds={}
    bounds['m']=(0,np.inf)
    bounds['A']=(0,np.inf)
    bounds['r']=(0,np.inf)
    bounds['c1']=(0,0.1)
    bounds['c2']=(-0.5,0.5)
    bounds['c3']=(-np.inf,np.inf)
    bounds['ch1']=(0,0.5)
    bounds['ch2']=(-0.5,0.5)
    bounds['di']=(0,1.0)
    bounds['dj']=(0,1.0)
    bounds['dk']=(0,2.0)
    bounds['vw_i']=(-15,15)
    bounds['vw_j']=(-15,15)
    bounds['kt']=(1.5,10.0)
    
    "scaler corresponds roughly to the power of ten of the parameter"
    "it does not have to though, it may be used to improve the grad descent"
    
    scalers={}
    scalers['m']=1.0
    scalers['A']=1.0
    scalers['r']=1.0
    scalers['c1']=1.0
    scalers['c2']=1.0
    scalers['c3']=1.0
    scalers['ch1']=1.0
    scalers['ch2']=1.0
    scalers['di']=1.0
    scalers['dj']=1.0
    scalers['dk']=1.0
    scalers['vw_i']=1.0
    scalers['vw_j']=1.0
    scalers['kt']=1.0
    
    
    
    metap={"model_motor_dynamics":model_motor_dynamics,
            "used_logged_v_in_model":used_logged_v_in_model,
            "with_ct3":with_ct3,
            "vanilla_force_model":vanilla_force_model,
            "structural_relation_idc1":structural_relation_idc1,
            "structural_relation_idc2":structural_relation_idc2,
            "fit_on_v":fit_on_v,
            "wind_signal":wind_signal,
            "assume_nul_wind":assume_nul_wind,
            "approx_x_plus_y":approx_x_plus_y,
            "di_equal_dj":di_equal_dj,
            "log_path":log_path,
            "base_lr":base_lr,
            "n_epochs":n_epochs,
            "nsecs":nsecs,
            "train_proportion":train_proportion,
            "[mass,Area,r,rho,kv_motor,pwmmin,pwmmax,U_batt,b1,c10,c20,c30,ch10,ch20,di0,dj0,dk0,vwi0,vwj0,kt0]":physical_params,
            "bounds":bounds}
    
    
    
    print(" META PARAMS \n")
    [print(i,":", metap[i]) for i in metap.keys()]
    
    
    
    # %%   ####### Saving function
    
    import os
    import json
    import datetime
    import time
    
    #generating a new log name if none is provided
    if log_name=="":
        log_name=str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        
    spath=os.path.join(os.getcwd(),save_dir_name,log_name)

    #on enlève les logs avec le même nom
    import shutil
    try:
        shutil.rmtree(spath)
    except:
        pass
    os.makedirs(spath)
    
    #on sauvegarde les paramètres de départ
    with open(os.path.join(spath,'data.json'), 'w') as fp:
        json.dump(metap, fp)
    
    import pandas as pd
    
    # la fonction saver va être utilisée souvent 
    
    def saver(name=None,save_path=os.path.join(os.getcwd(),"results_tests"),**kwargs):
        dname=str(int(time.time())) if (name is None) else name

        "le fonctionnement est simple:on prend tout ce qu'il y a dans **kwargs"
        "et  on le met dans un dictionnaire qu'on save dans un .json"
        
        D={}
        for i in kwargs:
            if type(kwargs[i])==dict:
                for j in kwargs[i]:
                    D[j]=kwargs[i][j]
            else:
                D[i]=kwargs[i].tolist() if isinstance(kwargs[i],np.ndarray) else kwargs[i]
        
        with open(os.path.join(save_path,'%s.json'%(dname)), 'w') as fp:
            json.dump(D, fp)
        return 0 
    
    
    # %%   ####### SYMPY PROBLEM 
    
    " le but de cette partie est de génerer une fonction qui renvoie : "
    " new_acc,new_v,omegas,sqerr_a,sqerr_v,Ja.T,Jv.T    , c'est à dire        "
    " l'acc pred, la vitesse pred, l'erreur quad sur acc, l'erreur quad sur v"
    " la jacobienne de l'erreur quad sur l'acc et la vitesse "
    
    import time
    
    t0=time.time()
    
    
    print("\nElapsed : %f s , Prev step time: -1 s \\ Generating first symbols ..."%(time.time()-t0))
    
    dt=symbols('dt',positive=True,real=True)
    m=symbols('m',reals=True,positive=True)
    
    m_scale=symbols('m_scale',real=True,positive=True)
    vw_i_scale,vw_j_scale=symbols('vw_i_scale,vw_j_scale',real=True,positive=True)
    kt_scale=symbols('kt_scale',real=True,positive=True)
    A_scale,r_scale=symbols('A_scale,r_scale',real=True,positive=True)
    c1_scale,c2_scale,c3_scale=symbols('c1_scale,c2_scale,c3_scale',real=True,positive=True)
    ch1_scale,ch2_scale=symbols('ch1_scale,ch2_scale',real=True,positive=True)
    di_scale,dj_scale,dk_scale=symbols('di_scale,dj_scale,dk_scale',real=True,positive=True)
    
    m_s=m*m_scale
    
    r1,r2,r3,r4,r5,r6,r7,r8,r9=symbols("r1,r2,r3,r4,r5,r6,r7,r8,r9",real=True)
    R=Matrix([[r1,r2,r3],
              [r4,r5,r6],
              [r7,r8,r9]])
    
    
    vlog_i,vlog_j,vlog_k=symbols("vlog_i,vlog_j,vlog_k",real=True)
    vpred_i,vpred_j,vpred_k=symbols("vpred_i,vpred_j,vpred_k",real=True)
    
    v_i,v_j,v_k=(vlog_i,vlog_j,vlog_k) if used_logged_v_in_model else (vpred_i,vpred_j,vpred_k)
    
    
    vw_i,vw_j=symbols('vw_i,vw_j',real=True)
    vw_i_s,vw_j_s=vw_i*vw_i_scale,vw_j*vw_j_scale
    
    v=Matrix([[v_i],
             [v_j],
             [v_k]])
    
    if not assume_nul_wind:
        va_NED=Matrix([[v_i-vw_i_s],
                        [v_j-vw_j_s],
                        [v_k]]) 
    else :
        va_NED=Matrix([[v_i],
                       [v_j],
                       [v_k]])
    
    va_body=R.T@va_NED
    
    k_vect=Matrix([[0],
                [0],
                [1.0]])
    
    
    "motor dynamics"
    t1=time.time()
    print("Elapsed : %f s , Prev step time: -1 s \\ Generating motor dynamics ..."%(t1-t0))
    
    kt=symbols('kt',real=True,positive=True)
    kt_s=kt*kt_scale
    omega_1,omega_2,omega_3,omega_4,omega_5,omega_6=symbols('omega_1,omega_2,omega_3,omega_4,omega_5,omega_6',real=True,positive=True)
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6=symbols('omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6',real=True,positive=True)
    
    omegas_c=Matrix([omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6])
    omegas=Matrix([omega_1,omega_2,omega_3,omega_4,omega_5,omega_6])
    
    omegas=omegas+dt*kt_s*(omegas_c-omegas) if model_motor_dynamics else omegas_c
    
    t2=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Solving blade model ..."%(t2-t0,t2-t1))
    
    "blade dynamics"
    
    b1=symbols('b1',real=True,positive=True)
    
    rho,A,omega,r=symbols('rho,A,omega,r',real=True,positive=True)
    A_s,r_s=A*A_scale,r*r_scale
    c1,c2,c3=symbols('c1,c2,c3',real=True)
    c1_s,c2_s,c3_s=c1*c1_scale,c2*c2_scale,c3*c3_scale
    
    ch1,ch2=symbols('ch1,ch2',real=True)
    ch1_s,ch2_s=ch1*ch1_scale,ch2*ch2_scale
    
    vi=symbols('eta',reals=True)
    
    v2=symbols('v2')
    v3=symbols('v3')
    
    if vanilla_force_model:
        T_sum=-c1_s*sum([ omegas[i]**2 for i in range(6)])*R@k_vect
        H_sum=0*k_vect
    else:
    
        T_BET=rho*A_s*r_s*omega*(c1_s*r_s*omega-c2_s*(vi-v3)) if not with_ct3 else rho*A_s*r_s*omega*(c1_s*omega*r_s-c2_s*(vi-v3))+c3_s*v2**2
        
        if structural_relation_idc1:
            T_BET=T_BET.subs(c2, b1*c1-2/b1)
        
        if structural_relation_idc2:
            T_BET=T_BET.subs(c1, c2/b1+2/b1*b1)
            
        T_MOMENTUM_simp=2*rho*A_s*vi*((vi-v3)+v2) if approx_x_plus_y else 2*rho*A_s*vi*(vi-v3)
        eq_MOMENTUM_simp=T_BET-T_MOMENTUM_simp
        
        eta=simplify(Matrix([solve(eq_MOMENTUM_simp,vi)[1]])).subs(v3,va_body[2,0]).subs(v2,sqrt(va_body[0,0]**2+va_body[1,0]**2))
        print(eta)

        etas=Matrix([eta.subs(omega,omegas[i]) for i in range(6)])
        
        def et(expr):
            return expr.subs(v3,va_body[2,0]).subs(v2,sqrt(va_body[0,0]**2+va_body[1,0]**2))
        
        T_sum=-simplify(sum([et(T_BET).subs(omega,omegas[i]).subs(vi,etas[i]) for i in range(6)]))*R@k_vect
        
        H_tmp=simplify(sum([r_s*omegas[i]*ch1_s+ch2_s*(etas[i]-va_body[2,0]) for i in range(6)]))
        H_sum=-rho*A_s*H_tmp*(va_NED-va_NED.dot(R@k_vect)*R@k_vect)
        # print(H_sum)
    t3=time.time()
    "liftdrag forces"
    print("Elapsed : %f s , Prev step time: %f s \\ Solving lifrdrag model ..."%(t3-t0,t3-t2))
    
    di,dj,dk=symbols('di,dj,dk',real=True,positive=True)
    di_s,dj_s,dk_s=di*di_scale,dj*dj_scale,dk*dk_scale
    D=diag(di_s,di_s,dk_s) if di_equal_dj else diag(di_s,dj_s,dk_s)
    Fa=-simplify(rho*A_s*va_NED.norm()*R@D@(R.T@va_NED))
    
    
    t35=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Solving Dynamics ..."%(t35-t0,t35-t3))
    
    g=9.81
    new_acc=simplify(g*k_vect+T_sum/m_s+H_sum/m+Fa/m_s)
    new_v=v+dt*new_acc
    
    t37=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Generating costs ..."%(t37-t0,t37-t35))
    
    alog_i,alog_j,alog_k=symbols("alog_i,alog_j,alog_k",real=True)
    alog=Matrix([[alog_i],[alog_j],[alog_k]])
    
    vnext_i,vnext_j,vnext_k=symbols("vnext_i,vnext_j,vnext_k",real=True)
    vnext_log=Matrix([[vnext_i],[vnext_j],[vnext_k]])
    
    
    err_a=Matrix(alog-new_acc)
    err_v=Matrix(vnext_log-new_v)
    
    cost_scaler_a=symbols('C_sa',real=True,positive=True)
    cost_scaler_v=symbols('C_sv',real=True,positive=True)
    
    sqerr_a=Matrix([1.0/cost_scaler_a*(err_a[0,0]**2+err_a[1,0]**2+err_a[2,0]**2)])
    sqerr_v=Matrix([1.0/cost_scaler_v*(err_v[0,0]**2+err_v[1,0]**2+err_v[2,0]**2)])
    
    # CI DESSOUS : on spécifie quelles variables sont les variables d'identif
   
    
    " constructing opti variables "
    " WARNING SUPER WARNING : the order must be the same as in id_variables !!!"
    
    id_variables_sym=[]
    
    id_variables_sym.append(m) if id_mass else None 
    
    id_variables_sym.append(A) if id_blade_geom_coeffs else None
    id_variables_sym.append(r) if id_blade_geom_coeffs else None
    
    id_variables_sym.append(c1) if id_blade_coeffs*(not structural_relation_idc2) or vanilla_force_model else None
    id_variables_sym.append(c2) if id_blade_coeffs*(not structural_relation_idc1) and (not vanilla_force_model) else None
    id_variables_sym.append(c3) if id_blade_coeffs*id_c3 else None
    id_variables_sym.append(ch1) if id_blade_coeffs*(not vanilla_force_model) else None
    id_variables_sym.append(ch2) if id_blade_coeffs*(not vanilla_force_model) else None
    
    
    id_variables_sym.append(di) if id_body_liftdrag_coeffs else None
    id_variables_sym.append(dj) if id_body_liftdrag_coeffs*(not di_equal_dj) else None
    id_variables_sym.append(dk) if id_body_liftdrag_coeffs else None
    
    id_variables_sym.append(vw_i) if id_wind else None
    id_variables_sym.append(vw_j) if id_wind else None
        
    id_variables_sym.append(kt) if id_time_const else None
    
    
    id_variables_sym=Matrix(id_variables_sym)
    
    print("\n ID VARIABLES:")
    print(id_variables_sym,"\n")
    
    
    
    t4=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Gathering identification parameters  ..."%(t4-t0,t4-t37))
    
    t5=time.time()
    
    Ja=sqerr_a.jacobian(id_variables_sym)
    Jv=sqerr_v.jacobian(id_variables_sym)
    
    t6=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Lambdification ..."%(t6-t0,t6-t5))
    
    
    
    X=(m,A,r,rho,
    b1,
    c1,c2,c3,
    ch1,ch2,
    di,dj,dk,
    vw_i,vw_j,
    kt,
    dt,cost_scaler_a,cost_scaler_v,
    vlog_i,vlog_j,vlog_k,
    vpred_i,vpred_j,vpred_k,
    alog_i,alog_j,alog_k,
    vnext_i,vnext_j,vnext_k,r1,r2,r3,r4,r5,r6,r7,r8,r9,
    omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6,
    m_scale,A_scale,r_scale,c1_scale,c2_scale,c3_scale,
    ch1_scale,ch2_scale,di_scale,dj_scale,dk_scale,
    vw_i_scale,vw_j_scale,kt_scale)
    
    
    
    Y=Matrix([new_acc,new_v,omegas,sqerr_a,sqerr_v,Ja.T,Jv.T])

    model_func=lambdify(X,Y, modules='numpy')
    
    t7=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Done ..."%(t7-t0,t7-t6))
    
    "cleansing memory"
    
    # très recommandé d'effacer les variables de sympy pour éviter les soucis 
    # dans le restes des fonctions
    
    del(dt,m,cost_scaler_a,cost_scaler_v,
    vlog_i,vlog_j,vlog_k,
    vpred_i,vpred_j,vpred_k,
    alog_i,alog_j,alog_k,
    vnext_i,vnext_j,vnext_k,
    vw_i,vw_j,
    kt,
    b1,
    c1,c2,c3,
    ch1,ch2,
    di,dj,dk,
    rho,A,r,r1,r2,r3,r4,r5,r6,r7,r8,r9,R,
    omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6)
    
    
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
    prep_data=prep_data.reset_index()
    
    data_prepared=prep_data[:len(prep_data)]
    
    
    
    for i in range(6):
        data_prepared['omega_c[%i]'%(i+1)]=(data_prepared['PWM_motor[%i]'%(i+1)]-pwmmin)/(pwmmax-pwmmin)*U_batt*kv_motor*2*np.pi/60
        
    "splitting the dataset into nsecs sec minibatches"
    
    print("SPLIT DATA...")
    
    if nsecs=='all':
        data_batches=[data_prepared]
        N_train_batches=1
        N_val_batches=0
        
    else:
        N_minibatches=round(data_prepared["t"].max()/nsecs) if nsecs >0 else  len(data_prepared)# 22 for flight 1, ?? for flight 2
        N_minibatches=N_minibatches if nsecs!='all' else 1

        data_batches=[i.drop(columns=[j for j in data_prepared.keys() if (("level" in j ) or ("index") in j) ]) for i in np.array_split(data_prepared, N_minibatches)]

        data_batches=[i.reset_index() for i in data_batches]
        
        N_train_batches=round(train_proportion*N_minibatches)
        N_val_batches=N_minibatches-N_train_batches
    print("DATA PROCESS DONE")
        
    # %%   ####### Identification Data Struct
    
    # On répartit les variables entre deux dicts: id_variables et non_id_variables
    # Cette étape est liée, par les booléens utilisés, à la premi_re étape
    
    non_id_variables={"m":mass,
                      "A":Area,
                      "r":r0,
                      "rho":rho0,
                      "b1":b10,
                      "c1":c10,
                      "c2":c20,
                      "c3":c30,
                      "ch1":ch10,
                      "ch2":ch20,
                      "di":di0,
                      "dj":dj0,
                      "dk":dk0,
                      "vw_i":vwi0,
                      "vw_j":vwj0,
                      "kt":kt0}
    id_variables={}
    
    if id_mass:
        id_variables['m']=mass  
        
    if id_blade_coeffs:
        if vanilla_force_model:
            id_variables['c1']=c10
        if not structural_relation_idc2:
            id_variables['c1']=c10
        if not structural_relation_idc1 and not vanilla_force_model:
            id_variables['c2']=c20
        if id_c3:
            id_variables['c3']=c30
        if not vanilla_force_model:
            id_variables['ch1']=ch10
            id_variables['ch2']=ch20
      
    if id_blade_geom_coeffs:
        id_variables['A']=Area
        id_variables['r']=r0
    
    if id_body_liftdrag_coeffs:
        id_variables['di']=di0
        if not di_equal_dj:
            id_variables['dj']=dj0
        id_variables['dk']=dk0
        
    if id_wind:
        id_variables['vw_i']=vwi0
        id_variables['vw_j']=vwj0
        if wind_signal:
            id_variables['vw_i']=vwi0*np.zeros(len(data_prepared))
            id_variables['vw_j']=vwj0*np.zeros(len(data_prepared)) 
            
    if id_time_const:
        id_variables['kt']=kt0
    
    "cleaning non_id_variables to avoid having variables in both dicts"
    rem=[]
    for i in non_id_variables.keys():
        if i in id_variables.keys():
            rem.append(i)
            
    for j in rem:
        del(non_id_variables[j])
    
    for i in id_variables.keys():
        id_variables[i]=id_variables[i]/scalers[i]
    

    # %%   ####### MODEL function
    
    # ici, on définit les fonctions que l'on appellera dans la partie 
    # optimisation. 
    
    
    import transforms3d as tf3d 
    import copy 
    
    def arg_wrapping(batch,id_variables,scalers,data_index,speed_pred_previous,omegas_pred):
        
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
        
        cost_scaler_v=1.0
        cost_scaler_a=1.0
    
        dt=min(batch['dt'][i],1e-2)

        vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        vpred_i,vpred_j,vpred_k=speed_pred_previous 
        alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]
        vnext_i,vnext_j,vnext_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        
        m=non_id_variables['m'] if 'm' in non_id_variables else id_variables['m']
        
        vw_i=non_id_variables['vw_i'] if 'vw_i' in non_id_variables else id_variables['vw_i']
        vw_j=non_id_variables['vw_j'] if 'vw_j' in non_id_variables else id_variables['vw_j']
        
        if wind_signal:
            vw_i,vw_j=vw_i[i],vw_j[i]
        
        
        kt=non_id_variables['kt'] if 'kt' in non_id_variables else id_variables['kt']
        b1=non_id_variables['b1'] if 'b1' in non_id_variables else id_variables['b1']
        c1=non_id_variables['c1'] if 'c1' in non_id_variables else id_variables['c1']
        c2=non_id_variables['c2'] if 'c2' in non_id_variables else id_variables['c2']
        c3=non_id_variables['c3'] if 'c3' in non_id_variables else id_variables['c3']
        ch1=non_id_variables['ch1'] if 'ch1' in non_id_variables else id_variables['ch1']
        ch2=non_id_variables['ch2'] if 'ch2' in non_id_variables else id_variables['ch2']
        di=non_id_variables['di'] if 'di' in non_id_variables else id_variables['di']
        dj=non_id_variables['dj'] if 'dj' in non_id_variables else id_variables['dj']
        dk=non_id_variables['dk'] if 'dk' in non_id_variables else id_variables['dk']
        rho=non_id_variables['rho'] if 'rho' in non_id_variables else id_variables['rho']
        A=non_id_variables['A'] if 'A' in non_id_variables else id_variables['A']
        r=non_id_variables['r'] if 'r' in non_id_variables else id_variables['r']
        
        R=tf3d.quaternions.quat2mat(np.array([batch['q[%i]'%(j)][i] for j in range(4)]))
        
        omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6=np.array([batch['omega_c[%i]'%(j)][i] for j in range(1,7,1)])
        omega_1,omega_2,omega_3,omega_4,omega_5,omega_6=omegas_pred
                
        m_scale=scalers['m']
        A_scale=scalers['A']
        r_scale=scalers['r']
        c1_scale,c2_scale,c3_scale=scalers['c1'],scalers['c2'],scalers['c3']
        ch1_scale,ch2_scale=scalers['ch1'],scalers['ch2']
        di_scale,dj_scale,dk_scale=scalers['di'],scalers['dj'],scalers['dk']
        vw_i_scale,vw_j_scale=scalers['vw_i'],scalers['vw_j']
        kt_scale=scalers['kt']
        
        X=(m,A,r,rho,
        b1,
        c1,c2,c3,
        ch1,ch2,
        di,dj,dk,
        vw_i,vw_j,
        kt,
        dt,cost_scaler_a,cost_scaler_v,
        vlog_i,vlog_j,vlog_k,
        vpred_i,vpred_j,vpred_k,
        alog_i,alog_j,alog_k,
        vnext_i,vnext_j,vnext_k,*R.flatten(),
        omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
        omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6,
        m_scale,A_scale,r_scale,c1_scale,c2_scale,c3_scale,
        ch1_scale,ch2_scale,di_scale,dj_scale,dk_scale,
        vw_i_scale,vw_j_scale,kt_scale)
        
        return X
    
    
    def pred_on_batch(batch,id_variables,scalers):
    
        "si n est la taille du batch"
        "cette fonction sert à appeler n fois la fonction lambdifiée"
        " de sympy "
        
        "on obtient n acc prédites, n vitesses prédites, n jacobiennes...."
        
    
        "batch est un dataframe"
        "id_variables sont les variables d'identification"
        "scalers sont les coefficients de mise à l'échelle"
        
        acc_pred=np.zeros((len(batch),3))
        speed_pred=np.zeros((len(batch),3))
        omegas=np.zeros((len(batch),6))    
            
        square_error_a=np.zeros((len(batch),1))    
        square_error_v=np.zeros((len(batch),1))    
        jac_error_a=np.zeros((len(batch),len(id_variables)))
        jac_error_v=np.zeros((len(batch),len(id_variables)))
        
        for i in batch.index:

            print("\r Pred on batch %i / %i "%(i,max(batch.index)), end='', flush=True)
    
            speed_pred_prev=speed_pred[i-1] if i>min(batch.index) else (batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i])
            omegas_pred=omegas[i-1] if i>0 else np.array([batch['omega_c[%i]'%(j)][i] for j in range(1,7,1)])
            X=arg_wrapping(batch,id_variables,scalers,i,speed_pred_prev,omegas_pred)
    
            Y=model_func(*X)
    
            acc_pred[i]=Y[:3].reshape(3,)
            speed_pred[i]=Y[3:6].reshape(3,)
            omegas[i]=Y[6:12].reshape(6,)
            square_error_a[i]=Y[12:13].reshape(1,)
            square_error_v[i]=Y[13:14].reshape(1,)
            jac_error_a[i]=Y[14:14+len(id_variables)].reshape(len(id_variables),)
            jac_error_v[i]=Y[14+len(id_variables):14+2*len(id_variables)].reshape(len(id_variables),)
            
    

        return acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v
    

    
    # %%   Gradient
    import random
    

    
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

        print(" --- Propagate %s grad : "%(log_name))
        

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

    def fun_cost_scipy(X,batch,scalers,writtargs):
        
        "cette fonction est la fonction que l'on utilisera si on choisit"
        "scipy comme algorithme de minmisation"
        "elle renvoie le coût, et la jacobienne"
        
        "X is the dict_to_X of id_variables"
        "dict reconstruction "
        
        n,k,val_sc_a,val_sc_v,total_sc_a,total_sc_v,t0,write_this_step,id_variables_base=writtargs


        id_var=X_to_dict(X,base_dict=id_variables_base)

        acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch,id_var,scalers)
        
        used_jac=jac_error_v if fit_on_v else jac_error_a
        used_err=square_error_v if fit_on_v else square_error_a

        #calcul de la jacobienne
        J=np.zeros(len(X))
        index_j=0
        for i,key in enumerate(id_var.keys()):
            if type(id_var[key]) is not np.ndarray:

                J[index_j]=np.mean(used_jac[:,i])
                index_j+=1
            else:
                J[index_j:index_j+len(id_var[key])]=used_jac[:,i]
                index_j+=len(id_var[key])



        C=np.mean(used_err,axis=0)
        
        print("\n %s------ Cost (in scipy minim): %f\n"%(log_name,C))
        
        "on repasse les valeurs dans leur valeur physique "
        realvals={}
        for i in id_var.keys():
            if ('vw' not in i):
                realvals[i]=id_var[i]*scalers[i]
            elif type(id_var[i]) is np.ndarray:
                realvals[i]=id_var[i][0]*scalers[i]
            else:
                realvals[i]=id_var[i]*scalers[i]
            
        if write_this_step:
            saver(name="epoch_%i_batch_%i_t_%f"%(n,k,time.time()-t0),save_path=spath,
            id_variables=realvals,
            train_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
            train_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
            val_sc_a=val_sc_a,
            val_sc_v=val_sc_v,
            total_sc_a=total_sc_a,
            total_sc_v=total_sc_v)
        if wind_signal and write_this_step:
            windsave_df=pd.DataFrame(data=np.array([id_var['vw_i'],id_var['vw_j']]).T,columns=['w_i','w_j'])
            nsave="WINDSIG_epoch_%i_batch_%i_t_%f"%(n,k,time.time())+".csv"
            windsave_df.to_csv(os.path.join(spath,nsave))
            del(windsave_df)
        
        return C,J
    
    # %% Train loop
    from scipy.optimize import minimize
    
    def train_loop(data_batches,id_var,n_epochs=n_epochs):
    
        #     on commence par copier le dict courant et randomiser, puis on sauvegarde
    
        id_variables=copy.deepcopy(id_var)
        print("Copy ...")
        temp_shuffled_batches=copy.deepcopy(data_batches)
        print("Done")
        print("INIT")
    
        total_sc_a=-1
        total_sc_v=-1
        print('\n###################################')
        print('############# Begin ident ###########')
        print("id_variables=",prepare_dict(id_variables,0),
              "train_sc_a=",-1,
              "train_sc_v=",-1,
              "val_sc_a=",-1,
              "val_sc_v=",-1,
              "total_sc_a=",total_sc_a,
              "total_sc_v=",total_sc_v,)
        print('###################################\n')
    
    
        "on repasse les valeurs dans leur valeur physique "
        realvals={}
        for i in id_variables.keys():
            realvals[i]=prepare_dict(id_variables,0)[i]*scalers[i]
    
        saver(name="start_",save_path=spath,
              id_variables=realvals,
              train_sc_a=-1,
              train_sc_v=-1,
              val_sc_a=-1,
              val_sc_v=-1,
              total_sc_a=total_sc_a,
              total_sc_v=total_sc_v)
        
        print("Entering training loop...")
        
        
        
        for n in range(n_epochs):
            "begin epoch"
            
            
            "train score correspond au score sur le dataset de d'entrainement"
            "val score correspond au score sur le dataset de validation "
            "total score correspond au score sur la totalité du dataset"
            
            train_sc_a,train_sc_v=0,0
            val_sc_a,val_sc_v=0,0
            total_sc_a,total_sc_v=0,0
            
            random.shuffle(temp_shuffled_batches)
            
            # on commence par les batches d'entrainement 
            
            for k,batch_ in enumerate(temp_shuffled_batches[:N_train_batches]):
                
                "si les batches sont tout petits, on génère beaucoup de résultats"
                "on ne veut pas tous els sauvegarder: ça prend de la place et "
                " ça peut provoquer des fuites de RAM "
                
                "on se donne donc des indexes de sauvegarde; en l'occurrence"
                " tous les 20% du dataset "
                
                save_indexes=np.arange(0,N_train_batches,max(N_train_batches//5,1))
                
                write_this_step=(k in save_indexes) or ns=='all'
                
                if fit_strategy not in ('custom_gradient','scipy'):
                    print(" ERROR WRONG FIT STRATEGY !!!!")
                    break
                            
                
                "si on a choisi d'utiliser le gradient custom pour minimiser"
                if fit_strategy=="custom_gradient":
                
                    acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables,scalers)
                    
                    temp_id_variables={}
                    for i in id_variables:
                        temp_id_variables[i]=id_variables[i]


                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            temp_id_variables[key_]=id_variables[key_][batch_.index]

                    new_id_variables=propagate_gradient(jac_error_v if fit_on_v else jac_error_a,temp_id_variables)


                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            tparr=id_variables[key_]
                            tparr[batch_.index]=new_id_variables[key_]
                            new_id_variables[key_]=tparr

                    id_variables={}
                    for i in new_id_variables:
                        id_variables[i]=new_id_variables[i]
                    
                    
                    train_sc_a+=np.sqrt(np.mean(square_error_a,axis=0))
                    train_sc_v+=np.sqrt(np.mean(square_error_v,axis=0))
                    print(" %s --- EPOCH : %i/%i || Train batch : %i/%i || PROGRESS: %i/%i [ta,tv]=[%f,%f]"%(log_name,n,
                                                                                                      n_epochs,
                                                                                                      k,N_train_batches,
                                                                                                      k,len(temp_shuffled_batches),
                                                                                                      np.sqrt(np.mean(square_error_a,axis=0)),
                                                                                                      np.sqrt(np.mean(square_error_v,axis=0))))
                    
                    realvals={}
                    for i in id_variables.keys():
                        if ('vw' not in i):
                            realvals[i]=id_var[i]*scalers[i]
                        elif type(id_var[i]) is np.ndarray:
                            realvals[i]=id_var[i][0]*scalers[i]
                        else:
                            realvals[i]=id_var[i]*scalers[i]


                    if write_this_step:

                        saver(name="epoch_%i_batch_%i"%(n,k),save_path=spath,
                          id_variables=realvals,
                          train_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                          train_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                          val_sc_a=val_sc_a,
                          val_sc_v=val_sc_v,
                          total_sc_a=total_sc_a,
                          total_sc_v=total_sc_v)
                        
                        if wind_signal:
                            print([i for i in id_variables],[id_variables[key] for key in id_variables.keys()])
                            windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                            nsave="WINDSIG_epoch_%i_batch_%i"%(n,k)+".csv"
                            windsave_df.to_csv(os.path.join(spath,nsave))
                            del(windsave_df)
                        gc.collect()
                        
                        
                        
                        
                elif fit_strategy=="scipy":
                    
                    "si on a choisi d'utiliser scipy pour minimiser"
                    
                    temp_id_variables={}
                    for i in id_variables:
                        temp_id_variables[i]=id_variables[i]
                        
                    "il faut s'y prendre comme ceci pour copier un dict,"
                    "faire dict_B=dict_A fait que si on modifie B on modifie A"
                    

                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            temp_id_variables[key_]=id_variables[key_][batch_.index]
                            

                    X_start=dict_to_X(temp_id_variables)
                    bnds=[bounds[i] for i in id_variables]


                    writtargs=[n,k,val_sc_a,val_sc_v,total_sc_a,total_sc_v,time.time(),write_this_step,temp_id_variables]
                       
                    sol_scipy=minimize(fun_cost_scipy,
                                       X_start,
                                       args=(batch_,scalers,writtargs),
                                        bounds=bnds,
                                        jac=True)#,options={"maxiter":1})
                    
                    new_id_variables=X_to_dict(sol_scipy["x"],base_dict=temp_id_variables)
                    
                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            tparr=id_variables[key_]
                            print(key_,tparr,new_id_variables[key_])
                            tparr[batch_.index]=new_id_variables[key_]
                            new_id_variables[key_]=tparr

                    id_variables={}
                    for i in new_id_variables:
                        id_variables[i]=new_id_variables[i]
                    
                    if wind_signal and write_this_step:
                        windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                        nsave="WINDSIG_epoch_%i_batch_%i"%(n,k)+".csv"
                        windsave_df.to_csv(os.path.join(spath,nsave))
                        del(windsave_df)
                        
                    if write_this_step:
                        gc.collect()
                        
                        
                    current_score=sol_scipy["fun"]
                    current_score_label="tv" if fit_on_v else "ta"
                    print(" %s --- EPOCH : %i/%i || Train batch : %i/%i || PROGRESS: %i/%i [%s]=[%f]"%(log_name,n,
                                                                                                      n_epochs,
                                                                                                      k,N_train_batches,
                                                                                                      k,len(temp_shuffled_batches),
                                                                                                      current_score_label,
                                                                                                      current_score))  


                    
                    
            train_sc_a/=N_train_batches
            train_sc_v/=N_train_batches
            
            
            # on itère sur les batches de validation
            if ns!="all" and N_val_batches>0:
                for k,batch_ in enumerate(temp_shuffled_batches[N_train_batches:]):
        
                    save_indexes=np.arange(0,N_val_batches,max(1,N_val_batches//10))
                    write_this_step=(k in save_indexes) or ns=='all'

                    acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables,scalers)
                    
                    val_sc_a+=np.mean(square_error_a,axis=0)
                    val_sc_v+=np.mean(square_error_v ,axis=0)   
                    print(" %s --- EPOCH : %i/%i || Eval batch : %i/%i || PROGRESS: %i/%i [va,vv]=[%f,%f]"%(log_name,
                                                                                                      n,n_epochs,
                                                                                                      k,N_val_batches,
                                                                                                      k+N_train_batches,len(temp_shuffled_batches),
                                                                                                      np.sqrt(np.mean(square_error_a,axis=0)),
                                                                                                      np.sqrt(np.mean(square_error_v,axis=0))))
                    realvals={}
                    for i in id_variables.keys():                        
                        if ('vw' not in i):
                            realvals[i]=id_var[i]*scalers[i]
                        elif type(id_var[i]) is np.ndarray:
                            realvals[i]=id_var[i][0]*scalers[i]
                        else:
                            realvals[i]=id_var[i]*scalers[i]

                    if write_this_step:
                        saver(name="epoch_%i_batch_%i"%(n,k+N_train_batches),save_path=spath,
                          id_variables=realvals,
                          train_sc_a=train_sc_a,
                          train_sc_v=train_sc_v,
                          val_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                          val_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                          total_sc_a=total_sc_a,
                          total_sc_v=total_sc_v)
    
                        if wind_signal:
                            windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                            nsave="WINDSIG_epoch_%i_batch_%i"%(n,k)+".csv"
                            windsave_df.to_csv(os.path.join(spath,nsave))
                        gc.collect()
                        
            if N_val_batches!=0:
                val_sc_a/=N_val_batches
                val_sc_v/=N_val_batches
    
            acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(data_prepared,id_variables,scalers)
            total_sc_a=np.sqrt(np.mean(square_error_a,axis=0))
            total_sc_v=np.sqrt(np.mean(square_error_v ,axis=0))
            
            realvals={}
            for i in id_variables.keys():
                if ('vw' not in i):
                    realvals[i]=id_var[i]*scalers[i]
                elif type(id_var[i]) is np.ndarray:
                    realvals[i]=id_var[i][0]*scalers[i]
                else:
                    realvals[i]=id_var[i]*scalers[i] 
                    
            print('\n###################################')
            print('############# END EPOCH ###########')
            print("id_variables=",realvals,
                  "train_sc_a=",train_sc_a,
                  "train_sc_v=",train_sc_v,
                  "val_sc_a=",val_sc_a,
                  "val_sc_v=",val_sc_v,
                  "total_sc_a=",total_sc_a,
                  "total_sc_v=",total_sc_v,)
            print('###################################\n')
    
            saver(name="epoch_%i"%(n),save_path=spath,
                  id_variables=realvals,
                  train_sc_a=train_sc_a,
                  train_sc_v=train_sc_v,
                  val_sc_a=val_sc_a,
                  val_sc_v=val_sc_v,
                  total_sc_a=total_sc_a,
                  total_sc_v=total_sc_v)
            
            if wind_signal:
                windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                nsave="WINDSIG_epoch_%i"%(n)+".csv"
                windsave_df.to_csv(os.path.join(spath,nsave))
        return 0 
          
    train_loop(data_batches,id_variables,n_epochs=n_epochs)
    
    return 0

from multiprocessing import Pool

if __name__ == '__main__':
    
    
    ns_range=['all']
    with_ct3_range=[True]
    vanilla_force_model_range=[False]
    assume_nul_wind_range=[False]
    di_equal_dj_range=[True,False]

    x_r=[]
    for i in ns_range:
        for j in with_ct3_range:
            for k in vanilla_force_model_range:
                for l in assume_nul_wind_range:
                    for p in di_equal_dj_range:
                        if not(j and k):
                            x_r.append([i,j,k,l,p])
                    
    x_r=[[i,j,k,l,p] for i in ns_range for j in with_ct3_range for k in vanilla_force_model_range for l in assume_nul_wind_range for p in di_equal_dj_range]
    

    # x_r.append([False, 'scipy', -1])

    # x_r.append([False, 'scipy', 1])
    # x_r.append([True, 'scipy',  1])
    
    # x_r.append([False, 'scipy', 5])
    # x_r.append([True, 'scipy',  5])
    
    # x_r.append([False, 'scipy', 'all'])
    # x_r.append([True, 'scipy',  'all'])

    
    print(x_r,len(x_r))

    pool = Pool(processes=8)
    alidhali=input('LAUNCH ? ... \n >>>>')
    pool.map(main_func, x_r)

