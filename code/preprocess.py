import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import netCDF4
import math
from netCDF4 import Dataset
from datetime import datetime,timedelta
from numpy import arange
import time
from readFile import read_ionFile
from readFile import read_ionFile_1h

"""
This program is used to preprocess all kinds of space weather parameters and unify their time resolution to 1 hour
Please note that the data source of this pre-processing is described in detail in the paper, 
and the data format of different sources may be different, 
so the following pre-processing procedures are only applicable to the data provided by the download website listed in the paper!!!
"""


def process_F10_7_SSN_Dst_ap(csvpath,outpath):
    "It is used to process.csv files where F10.7, SSN, dst, and ap reside"
    "csvpath such as: ./SpaceWeatherData/origin_data/OMNI2_H0_MRG1HR_95239.csv"
    "outpath such as: ./SpaceWeatherData/processed_data/processed_OMNI2_H0_MRG1HR_95239.csv"
    "Use the following example"
    """
    csvpath="./OMNI2_H0_MRG1HR_95239.csv"
    outpath="./processed_OMNI2_H0_MRG1HR_95239.csv"
    process_F10_7_SSN_Dst_ap(csvpath,outpath)
    """

    data = pd.read_csv(csvpath)
    data_origin=data.iloc[:,0:len(data)]
    for i in range(5):
        for j in range(len(data_origin)):
            if data_origin.iloc[j,i]==99999:
                data_origin.iloc[j,i]=data_origin.iloc[j-1,i]
            elif data_origin.iloc[j,i]==999.9:
                data_origin.iloc[j,i]=data_origin.iloc[j-1,i]   
            elif data_origin.iloc[j,i]==999:
                data_origin.iloc[j,i]=data_origin.iloc[j-1,i]    

    df=pd.DataFrame(data=data_origin,
                    columns=['TIME_AT_CENTER_OF_HOUR_yyyy-mm-ddThh:mm:ss.sssZ','DAILY_F10.7_','SSN','1-H_DST_nT','3-H_AP_nT'])
    df.to_csv(outpath, index=False)


def process_X_ray(filepath,outpath):
    "Multiple X-ray.csv files in a path are merged into one.csv. The X-ray sampling frequency is 1 minute, and the average value is taken every 60 minutes"
    '''
    #读取所有文件名
    Xray_filelist=get_obsFileAbPathList(filepath)
    Xray_data_list=[]
    for file in Xray_filelist:
        data=pd.read_csv(file)
        Xray_data=data.iloc[:,1]
        Xray_data=Xray_data.to_list()
        Xray_data_list=Xray_data_list+Xray_data
    df=pd.DataFrame(data=Xray_data_list,columns=['X-ray'])
    df.to_csv(outpath,index=False)
    '''
    
    #再将csv文件进行下采样处理
    new_Xray_data_list=[]
    count=0
    valid_count=0
    sum=0
    last_new_Xray=0
    data=pd.read_csv(filepath)
    Xray_data=data.iloc[:,0]
    print(len(Xray_data))
    for i in range(len(Xray_data)):
        t=Xray_data.iloc[i]
        t_num=float(t)
        count=count+1
        if (not pd.isna(t)) and (t_num!=0):
            valid_count=valid_count+1
            sum=sum+float(t)
        if count==60:
            if valid_count==0:
                new_Xray_data=last_new_Xray
            else:
                new_Xray_data=sum/valid_count
            new_Xray_data_list.append(new_Xray_data)
            last_new_Xray=new_Xray_data
            count=0
            valid_count=0
    df=pd.DataFrame(data=new_Xray_data_list,columns=['X-ray'])
    df.to_csv(outpath,index=False)
    print('ok')




def process_X_ray_nc(filepath,outpath):
    # Since xray files after 2017 are all in.NC format, we closeup another xray processing function. 
    # It should be noted that the netCDF read function Dataset() used in this function does not recognize the Chinese path, 
    # so the data file must be placed in the same directory as the.py when running
    '''
    # Use the following example
    # Example
    path='.\\2017-2022'
    outpath='D:\\processed_X_ray_2017_2022.csv'
    process_X_ray_nc(path,outpath)
    '''
    #get all Xrayfilepath
    Xray_filelist=get_obsFileAbPathList(filepath)
    X_ray_nc_list=[]
    X_ray_list_DownSampling=[]   #downsampling，1min to 1hour
    for file in Xray_filelist:
        nc_obj=Dataset(file)
        # for i in nc_obj.variables.keys():
        # print(i)
        xray = nc_obj['xrsb_flux_observed'][:].data
        num=len(xray)
        xray=xray[0:num]
        for i in range(0,num-1):
            if xray[i]<0 and i>1 :
                xray[i]=xray[i-1]
            X_ray_nc_list.append(xray[i])
    new_X_ray_nc_list = []
    sum_index = 0
    sum = 0
    for index in range(len(X_ray_nc_list)):
        sum += X_ray_nc_list[index]
        if sum_index == 59:
            sum /= 60
            new_X_ray_nc_list.append(sum)
            sum = 0
            sum_index = 0
        else:
            sum_index += 1
            continue


        sum_index += 1

    X_ray_list_DownSampling = new_X_ray_nc_list
    df=pd.DataFrame(data=X_ray_list_DownSampling,
                    columns=['X-ray'])
    df.to_csv(outpath,index=False)


def process_TEC_global_to_list(filepath):
    "The frequency of processing the TEC value of a single grid is 2h before October 19, 2014 (DoY:292), and 1h after that. Therefore, the TEC data of 2h should be divided into two 1h(global TEC statistics), and the final output shape is (days,24,71,73)."
    global_grid_TEC=[]
    TECfilelist=get_obsFileAbPathList(filepath)
    for TECfile in TECfilelist:
        year=TECfile[-3:-1]   
        doy=TECfile[-8:-4]  
        print(TECfile)
        if int(year) == 99:
            lltec = 0.1*read_ionFile(TECfile)
            for i in range(12):
                tmp = lltec[i]
                global_grid_TEC.append(tmp)
                global_grid_TEC.append(tmp)
            # global_grid_TEC.append(lltec)
            # global_grid_TEC.append(lltec)
        elif (int(year)<14) or (int(year)==14 and int(doy)<292):
            lltec = 0.1*read_ionFile(TECfile)
            for i in range(12):
                tmp = lltec[i]
                global_grid_TEC.append(tmp)
                global_grid_TEC.append(tmp)
            # global_grid_TEC.append(lltec)
            # global_grid_TEC.append(lltec)
        else:
            lltec = 0.1*read_ionFile_1h(TECfile)
            for i in range(24):
                tmp = lltec[i]
                global_grid_TEC.append(tmp)
    np.save('TEC_dataset.npy', global_grid_TEC)            
    return global_grid_TEC
        

def daysBetweenDates(date1, date2):
    "Get the number of days between two dates as shown in the following example"
    """
    # date1 = "2015-12-31"
    # date2 = "2008-01-01"
    # result = daysBetweenDates(date1, date2)
    # print(f"{date1} 和 {date2} 之间相隔：{result}天。")
    """
    y1, m1, d1 = date1.split("-")
    y2, m2, d2 = date2.split("-")
    cur_day = datetime(int(y1), int(m1), int(d1))
    next_day = datetime(int(y2), int(m2), int(d2))
    return abs((next_day - cur_day).days)


def get_obsFileAbPathList(obsPath):
    "Gets the absolute path for each observation file and returns a list"
    obsFileAbPath=[]
    for root,dirs,files in os.walk(obsPath):
        for f in files:
            obsFileAbPath.append(os.path.join(root,f))
    return obsFileAbPath



