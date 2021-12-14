#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:16:37 2021

@author: alex
"""

import pandas  as pd
import os
import numpy as np
import matplotlib.pyplot as plt
resdir=os.path.join(os.getcwd(),"res_avion_scipy_nosympy","FINAL_NEG_NOMEAN")

df_res=pd.concat([pd.read_csv(os.path.join(resdir,i)) for i in os.listdir(resdir) if "_" in i]).drop(columns=["Unnamed: 0"])

def gentypes(df):
    types=[]
    for i in range(len(df)):
        
        if not ('a_s'not in df ) and not(np.isnan(df['a_s'].values[i])):
            types.append('complex')
        elif not(np.isnan(df['a_0_v'].values[i])):
            types.append('simple')
        else:
            types.append('interm')
    return types

df_res.insert(1,'optim',gentypes(df_res))
# df_res['ct']*=1e5
# df_res=df_res.round(decimals=5)
# df_res['ct']*=1e-5

df_res['a_0'][df_res['a_0']==0.0]=[-0.07 for i in range(len([df_res['a_0']==0]))]
df_res['m'][df_res['fm']==True]=8.80*np.ones(len([df_res['fm']==True]))
df_res['k0']/=2.0
df_res['k1']/=2.0
df_res['k2']/=2.0


keps_keys=[
            'ct',
            'a_0',
            'cl1sa',
            'k0',
            'k1',
            'k2',
            'cd0sa',
            'cd1sa',
            'm'
            # 'a_0_v',
            # 'cl1sa_v',
            # 'k0_v',
            # 'k1_v',
            # 'k2_v',
            # 'cd0sa_v',
            # 'cd1sa_v'
            ]

# df_res=df_res.drop(columns=[i for i in df_res.keys() if i not in keps_keys])
df_res=df_res.sort_values(by=['optim'])

df_res.to_csv(os.path.join(resdir,"total.csv"))
df_res=df_res[df_res['optim']!='complex']

# df_res.hist(bins=100)

rgb=np.array([[df_res['fc']==True],
                [df_res['no_a_0']==True],
                [df_res['bnds']==True]]).T.astype(float)/2+0.25



df_res['rgb']=[i for i in rgb]

df=df_res[df_res['m']<20]
df=df[df['cl1sa']<10]
df=df[df['cl1sa']>-10]
df=df[df['cd1sa']<5]
df=df[df['cd1sa']>-5]
df=df[df['cd0sa']<0.5]
df=df[df['cd0sa']>-0.5]



bounds_ct= (0,1)
bounds_a_0= (-0.20,0.20) 
bounds_m=(5,15)
boundkeys=["ct",'a_0','m']

k0_0 = 0.0
k1_0 = 0.0
k2_0 = -7.5
a_0_0 = -0.07
ct_0 = 3.2726210849999994e-05
m_0 = 8.8

initkeys=["a_0","ct","m"]



# f,axes=plt.subplots(3,3)
# axes=axes.flatten()

# for i,key in enumerate(keps_keys):
#     ax=axes[i]
    
#     if key in boundkeys:
#         for i in eval("bounds_"+key):
#             ax.vlines(i,df['cost'].min(),df['cost'].max(),
#                       color="black",linestyle="--")
            
#     if key in initkeys:
#         ax.vlines(eval(key+"_0"),df['cost'].min(),df['cost'].max()
#                   ,color="purple",linestyle="--")
    
#     # ax.vlines(df[key].mean(),df['cost'].min(),df['cost'].max()
#     #           ,color="green",linewidth=2.0)
    
#     cols=np.array([i for i in df['rgb'].values])
    
#     ax.scatter(df[key],df['cost'],
#                c=cols,alpha=0.5)
    
#     ax.set_title(key),ax.grid()
#     ax.set_xlim(df[key].min()*1.2,df[key].max()*1.2)

from matplotlib import cm

# cd1sa vs ct

# f,axes=plt.subplots(1,2)
# tdf=df_res[df_res['fc']==False]
# tdf=tdf[abs(tdf['cd0sa'])<10]
# axes[0].scatter(tdf['ct'],tdf['cd0sa'],c=tdf['cost'])

# axes[0].grid()
# axes[0].set_xlabel(r'$c_T$'),axes[0].set_ylabel('$C_{D,0}^{sa}$')
# axes[1].scatter(tdf['ct'],tdf['cd1sa'],c=tdf['cost'])
# axes[1].grid()
# axes[1].set_xlabel(r'$c_T$'),axes[1].set_ylabel('$C_{D,1}^{sa}$')

# cbar=f.colorbar(cm.ScalarMappable(), ax=axes[1])
# cbar.set_label(r'$\frac{cost}{max(cost)}$', rotation=0, fontsize=15)

# alpha0 vs cost

# f,axes=plt.subplots(1,3)
# tdf=df_res[df_res['no_a_0']==False] 


# axes[0].scatter(tdf['a_0'],tdf['cost'],c=tdf['cost'])
# axes[0].grid()
# axes[0].set_xlabel(r'$\alpha_0$'),axes[0].set_ylabel('cost')

# tdf=df_res[df_res['cl1sa']<10]

# axes[1].scatter(tdf['cl1sa'],tdf['cost'],c=tdf['cost'])
# axes[1].grid()
# axes[1].set_xlabel(r'$C_{L,1}^{sa}$'),axes[1].set_ylabel('cost')


# axes[2].scatter(tdf['a_0'],tdf['cl1sa'],c=tdf['cost'])
# axes[2].grid()
# axes[2].set_xlabel(r'$\alpha_0$'),axes[2].set_ylabel(r'$C_{L,1}^{sa}$')

# cbar=f.colorbar(cm.ScalarMappable(), ax=axes[2])
# cbar.set_label(r'$\frac{cost}{max(cost)}$', rotation=0, fontsize=15)

# k0, k1, k2 vs cost

    
f,axes=plt.subplots(3,1)
# axes=[axes]
tdf=df_res[df_res['k0']!=0]

axes[0].scatter(tdf['k0'],tdf['cost'],c=tdf['cost'])
axes[0].set_xlabel(r'$\chi_d$'),axes[0].set_ylabel('cost'),axes[0].grid()

tdf=df_res[df_res['k1']!=0]

axes[1].scatter(tdf['k1'],tdf['cost'],c=tdf['cost'])
axes[1].set_xlabel(r'$\chi_l$'),axes[1].set_ylabel('cost'),axes[1].grid()

tdf=df_res[df_res['k2']!=0]

axes[2].scatter(tdf['k2'],tdf['cost'],c=tdf['cost'])
axes[2].set_xlabel(r'$\chi_{l,g}$'),axes[2].set_ylabel('cost'),axes[2].grid()


#comparaison no k1 vs no k0 

f,axes=plt.subplots(4,1)
axes=axes.flatten()
alldf=df_res[df_res['nok1']==False]
alldf=alldf[alldf["nok2"]==False]


tdf2=df_res[df_res['nok1']==True]   
nodf=tdf2[tdf2['nok2']==True]
tdf2=tdf2[tdf2['nok2']==False]

tdf1=df_res[df_res['nok2']==True]
tdf1=tdf1[tdf1['nok1']==False]

labels=['Both coeffs',r'Only $\chi_l$',r'Only $\chi_{l,g}$','None of both coeffs']

for temp_df,ax,lab in zip([alldf,tdf1,tdf2,nodf],axes,labels):
    
    ax.hist(temp_df['cost'],bins=200,weights=100*np.ones(len(temp_df)) / len(temp_df))
    ax.set_title(lab)
    ax.set_xlabel('cost'),ax.set_xlim(4,6),ax.set_ylabel('proportion (%)'),ax.grid()

#comparaison simple vs complexe

f,axes=plt.subplots(2,1)
axes=axes.flatten()
simdf=df_res[df_res['optim']=="simple"]
intdf=df_res[df_res['optim']!="simple"]

labels=['Modèle complet','Modèle simple']

for temp_df,ax,lab in zip([simdf,intdf],axes,labels):
    
    ax.hist(temp_df['cost'],bins=50,weights=100*np.ones(len(temp_df)) / len(temp_df))
    ax.set_title(lab)
    ax.set_xlabel('cost'),ax.set_xlim(4,None),ax.set_ylabel('proportion (%)'),ax.grid()


# cbar=f.colorbar(cm.ScalarMappable())
# cbar.set_label(r'$\frac{cost}{max(cost)}$', rotation=0, fontsize=15)
