#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:22:28 2020

@author: l3x
"""
import os
import pandas as pd
from pylab import *
from transforms3d import euler,quaternions
from scipy.interpolate import interp1d
from scipy.signal import resample
import pickle
import transforms3d as tf3d
import csv
import shutil, os

class Log():

    def __init__(self,ulg_path="",csv_path="",log_name=" ", 
    drone_type="plane",roll_gain=1.0,pitch_gain=1.0):

        self.ulg_path=ulg_path
        self.csv_path=csv_path
        self.DATA=-1
        self.drone_type=drone_type
        self.motor_gain_in_radsec=3500.0
        self.servo_gain=1.0
        self.log_name = log_name

        "gridsearch"
        self.roll_gain=roll_gain
        self.pitch_gain=pitch_gain
        return


    def log_to_csv(self):


        path=self.ulg_path
        print("Moving to...")
        os.chdir(path)
        
        if not 'donnees_brut' in os.listdir(path):
            os.mkdir("donnees_brut")
            
        print(path,"\n","Executing ulog2csv "+self.log_name+" -o donnees_brut")
        os.system('ulog2csv '+self.log_name+" -o donnees_brut")
        os.chdir('donnees_brut')
        
        print("Renaming files...")
        
        for file in os.listdir(os.getcwd()):
                if "actuator_outputs_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'actuators.csv'))
                if "battery_status_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'battery.csv'))
                if "vehicle_local_position_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'local_pos.csv'))
                if "gps_position_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'gps.csv'))
                if "actuator_controls_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'controls.csv'))
                if "angular_velocity_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'angular_vel.csv'))
                if "vehicle_attitude_groundtruth_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'attitude.csv'))
                elif "vehicle_attitude_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'attitude.csv'))
        os.chdir('../')

    def import_log(self):
        dpath=self.csv_path

        "check if there is already a DATA file in the directory"

        "READING CSV FILES"
        print(os.path.join(dpath))
        
        local_pos=pd.read_csv(os.path.join(dpath,"local_pos.csv"))
        act=pd.read_csv(os.path.join(dpath,"actuators.csv"))
        attitude=pd.read_csv(os.path.join(dpath,"attitude.csv"))
        ctrl=pd.read_csv(os.path.join(dpath,"controls.csv"))
        

        "preparing the interpolation"
        "the initial time for the interpolation is the latest first data for"
        "each file"

        tmin,tmax=max([min(i['timestamp']) for i in [local_pos,attitude,act]]),min([max(i['timestamp'].values) for i in [local_pos,attitude,act]])


        attitude=attitude.loc[(attitude['timestamp']>tmin )& (attitude['timestamp']<tmax)]
        local_pos=local_pos.loc[(local_pos['timestamp']>tmin )& (local_pos['timestamp']<tmax)]
        act=act.loc[(act['timestamp']>tmin )& (act['timestamp']<tmax)]
        ctrl=ctrl.loc[(ctrl['timestamp']>tmin )& (ctrl['timestamp']<tmax)]
        

        highest_freq_DF=[local_pos,attitude,act,ctrl][argmax([len(i) for i in [local_pos,attitude,act,ctrl]])]
        ref_timestamp=highest_freq_DF['timestamp'].values


        DATA=pd.DataFrame(data=ref_timestamp,columns=['t'])

        "reinterpolating all signals"

        "transforming quat into euler and mat"
        "not the same for sitl and real logs"
        
        "attitude is fetched from the attitude.csv file"

        att=asarray([euler.mat2euler(quaternions.quat2mat(attitude.values[i,4:8])) for i in range(len(attitude))]).reshape(-1,3)
        
        print("PLEASE CHECK THAT attitude.csv cols[4:8] corresponds to quat")
        R=array([quaternions.quat2mat(attitude.values[i,4:8]) for i in range(len(attitude))]).reshape(-1,9)

        for n_ctrl in range(8):
            ctrlinterp=interp1d(ctrl.timestamp.values,ctrl["control["+str(n_ctrl)+"]"].values)
            DATA['ctrl'+str(n_ctrl)]=ctrlinterp(clip(DATA['t'].values,min(ctrl.timestamp.values),max(ctrl.timestamp.values)))

        axinterp=interp1d(local_pos.timestamp.values,local_pos.ax.values)
        DATA['ax']=axinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        ayinterp=interp1d(local_pos.timestamp.values,local_pos.ay.values)
        DATA['ay']=ayinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        azinterp=interp1d(local_pos.timestamp.values,local_pos.az.values)
        DATA['az']=azinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vxinterp=interp1d(local_pos.timestamp.values,local_pos.vx.values)
        DATA['vx']=vxinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vyinterp=interp1d(local_pos.timestamp.values,local_pos.vy.values)
        DATA['vy']=vyinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vzinterp=interp1d(local_pos.timestamp.values,local_pos.vz.values)
        DATA['vz']=vzinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))
    
        vxinterp=interp1d(local_pos.timestamp.values,local_pos.x.values)
        DATA['x']=vxinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vyinterp=interp1d(local_pos.timestamp.values,local_pos.y.values)
        DATA['y']=vyinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vzinterp=interp1d(local_pos.timestamp.values,local_pos.z.values)
        DATA['z']=vzinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))



        attinterp=interp1d(attitude.timestamp.values,att.T)
        rsatt=attinterp(clip(DATA['t'].values,min(attitude.timestamp.values),max(attitude.timestamp.values)))


        Rinterp=interp1d(attitude.timestamp.values,R.T)

        rsR=Rinterp(clip(DATA['t'].values,min(attitude.timestamp.values),max(attitude.timestamp.values)))

        actinterp=interp1d(act.timestamp.values,act.values[:,2:10].T)
        ract=actinterp(clip(DATA['t'].values,min(act.timestamp.values),max(act.timestamp.values)))

        for i in range(8):
            DATA['act_'+str(i+1)]=ract[i]

        init_timestamp=DATA['t'].values[0]
        DATA['t']=(DATA['t']-DATA['t'].values[0])*1.0e-6


        DATA['roll'],DATA['pitch'],DATA['yaw']=rsatt
        DATA['init_timestamp ']=init_timestamp*ones(len(DATA))
        kv_motor=960.0
        pwmmin=1075.0
        pwmmax=1950.0

        DATA['dt']=gradient(DATA.t.values)


        DATA['ax_alt']=gradient(DATA.vx,DATA.t)
        DATA['ay_alt']=gradient(DATA.vy,DATA.t)
        DATA['az_alt']=gradient(DATA.vz,DATA.t)

        DATA['R']=list(rsR.T)

        local_speed=[matmul(DATA['R'].values[i].reshape((3,3)).T,array([DATA['vx'].values[i],DATA['vy'].values[i],DATA['vz'].values[i]]).reshape(3,)) for i in range(len(DATA))]
        local_acc=[matmul(DATA['R'].values[i].reshape((3,3)).T,array([DATA['ax_alt'].values[i],DATA['ay_alt'].values[i],DATA['az_alt'].values[i]]).reshape(3,)) for i in range(len(DATA))]
        local_speed=vstack(local_speed)
        local_acc=vstack(local_acc)
        DATA["vx_local"],    DATA["vy_local"],    DATA["vz_local"]=local_speed.T
        DATA["ax_local"],    DATA["ay_local"],    DATA["az_local"]=local_acc.T


        self.DATA=DATA
        DATA.to_csv('DATA.csv')
        
        return


    def logs_to_ctrl_input(self,step_number):
        if self.drone_type=="plane":
            ctrl_input={'motors_speed_sp':array([self.motor_gain_in_radsec*self.DATA['ctrl3'].values[step_number]]),
                        'wing_angle_sp':array([self.roll_gain*self.servo_gain*self.DATA['ctrl0'].values[step_number], # aileron right
                                               -self.roll_gain*self.servo_gain*self.DATA['ctrl0'].values[step_number], # aileron left
                                               self.pitch_gain*self.servo_gain*self.DATA['ctrl1'].values[step_number], # elevon
                                               self.servo_gain*self.DATA['ctrl2'].values[step_number], # rudder
                                               self.servo_gain*self.DATA['ctrl4'].values[step_number], # airbrake
                                               self.servo_gain*self.DATA['ctrl4'].values[step_number]]), # airbrake
                        'motors_tilt_sp':-array([pi/2])}

        return ctrl_input
    
    def data_to_dic(self):
        # self.DATA= pickle.load(open(self.csv_path+"/DATA", "rb"))
        f = self.DATA
        dic_data={}
        
        q = [tf3d.quaternions.mat2quat(f.R.values[i]) for i in range(len(f.t))]

        dic_data['t']=f.t.values
        
        dic_data['acc[0]']=f.ax.values
        dic_data['acc[1]']=f.ay.values
        dic_data['acc[2]']=f.az.values

        dic_data['speed[0]']=f.vx.values
        dic_data['speed[1]']=f.vy.values
        dic_data['speed[2]']=f.vz.values
    
        dic_data['pos[0]']=f.x.values
        dic_data['pos[1]']=f.y.values
        dic_data['pos[2]']=f.z.values
        
        dic_data["q[0]"]= [q[j][0] for j in range(len(q))]
        dic_data["q[1]"] = [q[j][1] for j in range(len(q))]
        dic_data["q[2]"] = [q[j][2] for j in range(len(q))]
        dic_data["q[3]"] = [q[j][3] for j in range(len(q))] 

        dic_data['forces[0]']=f.ax.values * 8.5
        dic_data['forces[1]']=f.ay.values * 8.5
        dic_data['forces[2]']=f.az.values * 8.5
        
        dic_data['joystick[0]']=f.ctrl0.values * 250
        dic_data['joystick[1]']=f.ctrl1.values * 250
        dic_data['joystick[2]']=f.ctrl2.values * 250 
        dic_data['joystick[3]']=(f.ctrl3.values-0.5)*2 * 250

        dic_data['PWM_motor[1]']=f.act_1.values 
        dic_data['PWM_motor[2]']=f.act_2.values 
        dic_data['PWM_motor[3]']=f.act_3.values 
        dic_data['PWM_motor[4]']=f.act_4.values 
        dic_data['PWM_motor[5]']=f.act_5.values 
        dic_data['PWM_motor[6]']=f.act_6.values        
        dic_data['PWM_motor[7]']=f.act_7.values        
        dic_data['PWM_motor[8]']=f.act_8.values        

        dic_data['takeoff']=[1 if f.z.values[i]<-2 else 0 for i in range(len(f.t))]

        with open(os.path.join(self.ulg_path+"/log_real.csv"),'a',newline='') as log:
            spamwriter = csv.writer(log)
            spamwriter.writerow(dic_data.keys())
            for j in range(len(f.t)):
                if dic_data['takeoff'][j]>-1:
                    row = [values[j] for values in dic_data.values()]
                    spamwriter.writerow(row)


"creating and/or  cleaning the ./logs/csv directory "
if "csv" not in os.listdir(os.path.join(os.getcwd(),"logs")):
    os.mkdir(os.path.join(os.getcwd(),"logs","csv"))
    
else:
    for f in os.listdir(os.path.join(os.getcwd(),"logs","csv")):
        os.remove(os.path.join(os.path.join(os.getcwd(),"logs","csv"), f))
    
target_log_path=os.path.join(os.getcwd(),"logs","target")

log=[i for i in os.listdir(target_log_path)
     if "ulg" in i][0][:-4]

l=Log(ulg_path=os.path.join(os.getcwd(),"logs","target"),
csv_path=os.path.join(os.getcwd(),"logs","csv"), 
log_name=log+'.ulg')

"log2csv renames the "
l.log_to_csv()
l.import_log()
l.data_to_dic()
