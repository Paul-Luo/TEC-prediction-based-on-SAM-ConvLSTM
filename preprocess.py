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
本程序用于预处理各类空间天气参数
"""


def process_F10_7_Dst_Ap(csvpath,outpath):
    "专用于处理F10.7、dst、Ap所在的.csv文件"
    "csvpath比如为./SpaceWeatherData/origin_data/OMNI2_H0_MRG1HR_95239.csv"
    "outpath比如为./SpaceWeatherData/processed_data/processed_OMNI2_H0_MRG1HR_95239.csv"
    "使用示例如下"
    """
    csvpath="./SpaceWeatherData/origin_data/OMNI2_H0_MRG1HR_95239.csv"
    outpath="./SpaceWeatherData/processed_data/processed_OMNI2_H0_MRG1HR_95239.csv"
    process_F10_7_Dst_Ap(csvpath,outpath)
    """

    data = pd.read_csv(csvpath)
    data_origin=data.iloc[:,0:len(data)]
    for i in range(4):
        for j in range(len(data_origin)):
            if data_origin.iloc[j,i]==99999:
                data_origin.iloc[j,i]=data_origin.iloc[j-1,i]
            elif data_origin.iloc[j,i]==999.9:
                data_origin.iloc[j,i]=data_origin.iloc[j-1,i]   
            elif data_origin.iloc[j,i]==999:
                data_origin.iloc[j,i]=data_origin.iloc[j-1,i]    

    df=pd.DataFrame(data=data_origin,
                    columns=['TIME_AT_CENTER_OF_HOUR_yyyy-mm-ddThh:mm:ss.sssZ','DAILY_F10.7_','1-H_DST_nT','3-H_AP_nT'])
    df.to_csv(outpath, index=False)
    

def process_EUV(EUVpath,outpath):
     "专用于处理EUV文件,包括08-15年的各个文件夹,最后处理为一个.csv文件进行保存,使用示例如下"
     """
    #下面3行是EUV数据生成代码
    # path='D:\恢复\CELIAS_SEM_15sec_avg'
    # outpath='D:\\恢复\\processed_EUV.csv'
    # process_EUV(path,outpath) 
    #下面6行是EUV数据测试代码
    # path='D:\\恢复\\processed_EUV.csv'
    # data=pd.read_csv(path)
    # data=data.iloc[:,0]
    # print("length:",len(data))
    # data.plot()
    # plt.show()
     """
     filefolderlist=get_obsFileAbPathList(EUVpath)
     #取2008年-2015年期间的数据[4207:7128]（1999-2007年底为[971:4206]）
    #  filefolderlist=filefolderlist[971:4206]
     #遍历每天的EUV数据
     EUV=[]
     last_data=0
     for file in filefolderlist:
        with open(file,'r') as file_content:
            temp=file_content.readlines()
            #直接截取EUV数据部分，从第47行到末尾
            EUV_content=temp[46:len(temp)]
            #频率为15s，则一个EUV文件理论上有5760个数据，但实际并没有（一般大约5717个），因此此处进行数据扩充，need_num为需扩充的数据量（比如5760-5717=43）,将最后一个数据进行填充即可
            total_num=len(EUV_content)
            need_num=5760-total_num
            for line in EUV_content:
                line=line.split()
                #取26-34nm波段的通量
                EUV.append(float(line[12]))
                last_data=float(line[12])
            if need_num>0:
                for k in range(need_num+2):
                    EUV.append(last_data)
            # print("OK")
    #  for k in range(158):
    #     EUV.append(last_data)
     #原始频率为15s，现进行下采样，使其频率为1h，即每4*60=240个数据取其平均值作为最后数值
     EUV_DownSampling=[]
     for k in range(0,len(EUV),240):
        temp_sum=0
        for j in range(240):
            if (k+j)==(len(EUV)-1):
                temp_sum=temp_sum+EUV[k+j]
                break
            else:
                temp_sum=temp_sum+EUV[k+j]
        EUV_DownSampling.append(temp_sum/240)
        last_data=temp_sum/240
    #  for k in range(1223):
    #     EUV_DownSampling.append(last_data)
     df=pd.DataFrame(data=EUV_DownSampling,
                    columns=['EUV'])
     df.to_csv(outpath, index=False)


def process_X_ray(filepath,outpath):
    "将路径下多个X-ray的.csv文件融合为一个.csv,由于X-ray采样频率为1min,所以进行下采样,每60min就取一次平均值,示例代码如下"
    """
    path='D:\\恢复\\X-ray'
    outpath='D:\\恢复\\processed_X_ray.csv'
    process_X_ray(path,outpath)
    """
    #读取所有文件名
    Xray_filelist_unordered=get_obsFileAbPathList(filepath)
    #由于下载的Xray文件名开头并不是固定的(有g10、g11、g15等),所以上一步读取出来的文件名并没有按照时间顺序进行排列，故下面进行排序操作
    Xray_filelist=[]
    filedate_str_list=[]
    for file in Xray_filelist_unordered:
        date_begin_index=file.find('1m_')
        date_begin_index=date_begin_index+3
        date_end_index=date_begin_index+8
        filedate_str=file[date_begin_index:date_end_index]    #每个文件名的第date_begin_index到第date_end_index位是日期
        filedate_str=int(filedate_str)   #转为int类型，方便后续排序
        filedate_str_list.append(filedate_str)
    #对列表中的日期元素进行排序
    filedate_str_list.sort()
    for date in filedate_str_list:
        # temp=[involved(str(date),out) for out in Xray_filelist_unordered]
        for unordered_date in Xray_filelist_unordered:
            if str(date) in unordered_date:
                Xray_filelist.append(unordered_date)
                break
    X_ray_list=[]
    X_ray_list_DownSampling=[]
    total=0
    for filename in Xray_filelist:
        #注意g10、g11、g14、g15开头的文件里的格式稍有区别，g10和g11一样的，g15和g14不一样，下面读取文件里的数据时需要注意格式问题，这里用flag做标记
        flag=0
        if 'g10' in filename or 'g11' in filename:
            flag=1
        elif 'g15' in filename or 'g14' in filename:
            flag=2
        #找到并记录x-long-ray数据起始行
        begin_num=0
        with open(filename,'rb') as f:
            line=f.readline()
            while line:
                if (b'time_tag' in line and b'xs' in line and b'xl' in line) or (b'time_tag' in line and b'A_QUAL_FLAG' in line and b'B_NUM_PTS' in line):
                    begin_num=begin_num+1
                    break
                begin_num=begin_num+1
                line=f.readline()
        with open(filename,'rb') as f:
            content=f.readlines()
            content_=content[begin_num:len(content)]
            total=total+(len(content)-begin_num+1)
            for line in content_:
                line=line.split(b',')
                Xray=line[2] if flag==1 else line[6]
                #Xray文件里缺少不少missing值(文件里用-99999来表示)，所以对于missing值统一替换为通常值7.18e-09
                # Xray=b'7.18e-09' if Xray==b'-99999.0\r\n' or Xray==b'-99999\r\n' else Xray
                Xray=b'7.18e-09' if float(Xray)<-9999 else Xray
                X_ray_list.append(float(Xray))
    for k in range(0,len(X_ray_list),60):
        temp_sum=0
        for j in range(60):
            temp_sum=temp_sum+X_ray_list[k+j]
        X_ray_list_DownSampling.append(temp_sum/60)
    #填充一下
    for k in range(672):
        X_ray_list_DownSampling.append(float(7.18e-09))
    df=pd.DataFrame(data=X_ray_list_DownSampling,
                    columns=['X-ray'])
    df.to_csv(outpath,index=False)


def process_X_ray_2(filepath,outpath):
    #先生成总的Xray的csv文件
    '''
    #读取所有文件名
    Xray_filelist=get_obsFileAbPathList(filepath)
    Xray_data_list=[]
    for file in Xray_filelist:
        data=pd.read_csv(file)
        Xray_data=data.iloc[:,1]
        Xray_data=Xray_data.to_list()
        # time=file[-6:-4]
        # right_num=0
        # if time=='31':  
        #     right_num=44640
        # elif time=='30':
        #     right_num=43200
        # elif time=='29':
        #     right_num=41760
        # elif time=='28':
        #     right_num=40320
        # if right_num!=len(Xray_data):
        #     print('filename:'+file)
        #     print(len(Xray_data))
        # print('filename:'+file)
        # print(len(Xray_data))
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
    "由于2017年之后的xray文件都为.nc格式,所以特写另一个xray处理函数,注意本函数中用到的netCDF的读取函数Dataset()不识别中文路径,所以运行时须将数据文件放于本.py同目录下"
    '''
    #示例代码
    path='.\\2017-2022'
    outpath='D:\\恢复\\processed_X_ray_2017_2022.csv'
    process_X_ray_nc(path,outpath)
    '''
    #读取所有文件名
    Xray_filelist=get_obsFileAbPathList(filepath)
    X_ray_nc_list=[]
    X_ray_list_DownSampling=[]   #下采样，原nc文件频率为1min，现改为1hour
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

    # for k in range(0,len(X_ray_nc_list),60):
    #     X_ray_list_DownSampling.append(X_ray_nc_list[k])
    X_ray_list_DownSampling = new_X_ray_nc_list
    df=pd.DataFrame(data=X_ray_list_DownSampling,
                    columns=['X-ray'])
    df.to_csv(outpath,index=False)

def process_X_ray_nc_2(filepath,outpath):
    "由于2017年之后的xray文件都为.nc格式,所以特写另一个xray处理函数,注意本函数中用到的netCDF的读取函数Dataset()不识别中文路径,所以运行时须将数据文件放于本.py同目录下"
    #PS：这个process_X_ray_nc_2不同于上面的process_X_ray_nc函数，process_X_ray_nc_2是将2017年及以后的.nc数据（一年）转为12个.csv数据（月），中间不涉及取平均数和下采样之类的操作
    '''
    #示例代码
    path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\2016_2022\\2017-2022\\2022'
    outpath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\2016_2022\\2017-2022\\2022\\g16_xrs_1m_20220101_20221231.csv'
    process_X_ray_nc_2(path,outpath)
    '''
    #读取所有文件名
    Xray_filelist=get_obsFileAbPathList(filepath)
    xray=[]
    time=[]
    for file in Xray_filelist:
        nc_obj=Dataset(file)
        # for i in nc_obj.variables.keys():
        #     print(i)
        xray = nc_obj['xrsb_flux_observed'][:].data
    time=get_date_list('2022-01-01 00:00:00', '2022-12-31 23:59:59')
    dic_data={"time_tag":time,
            "B_AVG":xray}
    df=pd.DataFrame(dic_data)
    # df=pd.DataFrame(data=xray,
    #                 columns=['B_AVG'])
    df.to_csv(outpath,index=False)

def draw_picture(begin_date,end_date):
    "画ap,Dst,EUV,F10.7,Kp,X-ray图"
    




def process_TEC_single(filepath,outpath):
    "对单网格TEC值进行处理,频率在2014年10月19日(DoY:292)之前为2h,之后为1h,所以处理2h的TEC数据时,需要分为2个1h(先统计一个固定格网点的从2008-2015年的TEC数值,比如北京?[40N,115E])"
    BJ_grid_TEC=[]
    TECfilelist=get_obsFileAbPathList(filepath)
    for TECfile in TECfilelist:
        year=TECfile[-3:-1]   #从文件名获取年
        doy=TECfile[-8:-4]  #从文件名获取年积日
        # lltec = 0.1*read_ionFile(TECfile) if (int(year)<14) or (int(year)==14 and int(doy)<292) else 0.1*read_ionFile_1h(TECfile)
        flag=0
        print(TECfile)
        if (int(year)<14) or (int(year)==14 and int(doy)<292):
            lltec = 0.1*read_ionFile(TECfile)
            flag=1
        else:
            lltec = 0.1*read_ionFile_1h(TECfile)
            flag=2
        
        temp=lltec[:,19,59]    #lltec[:,19,59]代表北京所在格网点的TEC值    
        temp=(temp.transpose()).tolist()
        for item in range(len(temp)):
            #如果频率在2014年10月19日(DoY:292)之前,则需要分为2个1h
            if flag==1:
                BJ_grid_TEC.append(temp[item])  
                BJ_grid_TEC.append(temp[item])  
            else:
                BJ_grid_TEC.append(temp[item])  
    for index in range(24):
        BJ_grid_TEC.append('5')  
    df=pd.DataFrame(data=BJ_grid_TEC,
                    columns=['BJ_TEC'])
    df.to_csv(outpath,index=False)


def process_TEC_global_to_list(filepath):
    "对单网格TEC值进行处理,频率在2014年10月19日(DoY:292)之前为2h,之后为1h,所以处理2h的TEC数据时,需要分为2个1h(统计全球global TEC),最后输出的shape为(days,24,71,73)"
    global_grid_TEC=[]
    TECfilelist=get_obsFileAbPathList(filepath)
    for TECfile in TECfilelist:
        year=TECfile[-3:-1]   #从文件名获取年
        doy=TECfile[-8:-4]  #从文件名获取年积日
        print(TECfile)
        if (int(year)<14) or (int(year)==14 and int(doy)<292):
            lltec = 0.1*read_ionFile(TECfile)
            global_grid_TEC.append(lltec)
            global_grid_TEC.append(lltec)
        else:
            lltec = 0.1*read_ionFile_1h(TECfile)
            global_grid_TEC.append(lltec)
    return TECfilelist
        

def daysBetweenDates(date1, date2):
    "获取两个日期之间的天数间隔，使用示例如下"
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
    "获取每个观测文件的绝对路径，返回一个列表"
    obsFileAbPath=[]
    for root,dirs,files in os.walk(obsPath):
        for f in files:
            obsFileAbPath.append(os.path.join(root,f))
    return obsFileAbPath

# 获得begin_date到end_date之间的时间列表（以分钟为间隔）
def get_date_list(begin_date, end_date):
    "a=get_date_list('2018-01-01 00:00:00', '2018-02-28 00:00:00')"
    dates = []
    # Get the time tuple : dt
    dt = datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        dt += timedelta(minutes=1)
        date = dt.strftime("%Y-%m-%d %H:%M:%S")
   
   
# 获得begin_date到end_date之间的时间列表,以interval为间隔(interval_type为时间类型,分为second,minute,hour,day)
def get_date_list(begin_date, end_date,interval,interval_type):
    "a=get_date_list('2018-01-01 00:00:00', '2018-02-28 00:00:00',5,minute)"
    # begin_date='2018-01-01 '+begin_date
    # end_date='2018-01-01 '+end_date
    dates = []
    # Get the time tuple : dt
    dt = datetime.strptime(begin_date, "%Y-%m-%d")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        if interval_type=='minute':
            dt += timedelta(minutes=interval)
        elif interval_type=='second':
            dt += timedelta(seconds=interval)
        elif interval_type=='hour':
            dt += timedelta(hours=interval)   
        elif interval_type=='day':
            dt += timedelta(days=interval)   
        date = dt.strftime("%Y-%m-%d")
    return dates


def query_date_list(input_dst_query_,input_kp_query_,input_xray_query_):
    date_list=get_date_list('1999-01-01', '2022-12-31',1,interval_type='day')
    file_data = pd.read_csv('D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\total_event_v2.csv')
    date=file_data.iloc[:,0]
    dst=file_data.iloc[:,1]
    kp1=file_data.iloc[:,3]
    kp2=file_data.iloc[:,4]
    kp3=file_data.iloc[:,5]
    kp4=file_data.iloc[:,6]
    kp5=file_data.iloc[:,7]
    kp6=file_data.iloc[:,8]
    kp7=file_data.iloc[:,9]
    kp8=file_data.iloc[:,10]
    Xray_C=file_data.iloc[:,12]
    Xray_M=file_data.iloc[:,13]
    Xray_X=file_data.iloc[:,14]

    #待查询变量及值
    # input_dst_query='5'
    # input_kp_query='5'
    # input_xray_query='y'
    input_dst_query=input_dst_query_
    input_kp_query=input_kp_query_
    input_xray_query=input_xray_query_

    dst_query=input_dst_query.split('&')
    dst_query_list=[]
    for i in range(len(dst_query)):
        dst_query_list.append(int(dst_query[i]))

    kp_query=input_kp_query.split('&')
    kp_query_list=[]
    for i in range(len(kp_query)):
        kp_query_list.append(int(kp_query[i]))

    xray_query=input_xray_query.split('&')
    xray_query_list=[]
    for i in range(len(xray_query)):
        xray_query_list.append(xray_query[i])

    #先查dst
    dst_date_list=[]
    for i in range(len(dst_query_list)):
        cur_dst_query=dst_query_list[i]
        for j in range(len(dst)):
            if cur_dst_query==dst[j]:
                dst_date_list.append(date_list[j])


    kp_list=[kp1,kp2,kp3,kp4,kp5,kp6,kp7,kp8]
    kp_list=np.array(kp_list,dtype=int)
    #再查Kp
    kp_date_list=[]
    for i in range(len(kp_query_list)):
        cur_kp_query=kp_query_list[i]
        for j in range(len(kp1)):
            if cur_kp_query in kp_list[:,j]:
                kp_date_list.append(date_list[j])
    #对结果列表进行去重
    kp_date_list=list(set(kp_date_list))

    #再查Xray
    xray_date_list=[]
    for i in range(len(xray_query_list)):
        cur_xray_query=xray_query_list[i]
        if cur_xray_query=='C':
            for j in range(len(Xray_C)):
                if Xray_C[j]>0:
                    xray_date_list.append(date_list[j])
        if cur_xray_query=='M':
            for j in range(len(Xray_M)):
                if Xray_M[j]>0:
                    xray_date_list.append(date_list[j])
        if cur_xray_query=='X':
            for j in range(len(Xray_X)):
                if Xray_X[j]>0:
                    xray_date_list.append(date_list[j])
    #对结果列表进行去重
    xray_date_list=list(set(xray_date_list))


    # #对三个结果列表取并集
    # temp_list=list(set(dst_date_list).union(set(kp_date_list)))
    # fianl_list=list(set(temp_list).union(set(xray_date_list)))
    # return fianl_list

    #对三个结果列表取交集
    temp_list=[i for i in dst_date_list if i in kp_date_list]
    fianl_list=[i for i in temp_list if i in xray_date_list]
    return fianl_list


def get_timelist_sorted(date):
    return datetime.datetime.strptime(date,"%Y-%m-%d").timestamp()


if __name__ == '__main__':
    # date1 = "2019-11-01"
    # date2 = "2020-01-31"
    # result = daysBetweenDates(date1, date2)
    # print(f"{date1} 和 {date2} 之间相隔：{result}天。")


    # path='D:\\tempdata'
    # lltec_list = []
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         path_ = os.path.join(root, file)
    #         lltec = read_ionFile_1h(path_) * 0.1
    #         lltec /= 150
    #         # lltec_list.append(lltec)
    #         # lltec_list += lltec
    #         lltec_list = np.concatenate((lltec_list,lltec)) if lltec_list!=[] else lltec
    
    # np.save(path+'\\TEC_20230422_20230425.npy',lltec_list)


    # path='D:\\tempdata'
    # outpath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\TEC格网预测\\空间天气参数数据\\2023\\processed_X_ray_2023.csv'
    # process_X_ray_nc(path,outpath)


    path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\EUV\\2023'
    outpath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\EUV\\2023\\processed_20230421_20230425_EUV.csv'
    process_EUV(path,outpath) 




    print('ok')



#     Kp_event_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Kp\\processed_Kp_3h_1999_2022.csv'
#     Dst_event_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Dst\\processed_Dst_1h_1999_2022.csv'
#     Xray_event_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray'
#     out_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data'

#     # Dst_file = pd.read_csv(Dst_event_path)
#     # Dst_data = Dst_file.iloc[:, 1]  
#     # Dst_date = Dst_file.iloc[:, 0]   # 获取时间信息
#     # Dst_level=Dst_file.iloc[:, 2] 

#     # Kp_file = pd.read_csv(Kp_event_path)
#     # Kp_data = Kp_file.iloc[:, 1]  
#     # Kp_date = Kp_file.iloc[:, 0]   # 获取时间信息
#     # Kp_level=Kp_file.iloc[:, 2] 

#     # Xray_filelist=get_obsFileAbPathList(Xray_event_path)
#     # for file in Xray_filelist:
#     #     level_list=[]
#     #     if 'processed_' in file:
#     #         Xray_file=pd.read_csv(file)
#     #         Xray_data = Xray_file.iloc[:, 1]  
#     #         Xray_time = Xray_file.iloc[:, 0]   # 获取时间信息
#     #         Xray_level = Xray_file.iloc[:, 2]

#     date_list=get_date_list('1999-01-01', '2022-12-31',1,interval_type='day')
#     Dst_level_list=[]
#     Kp_level_list=[]
#     # for i in range(len(date_list)):
#     #     Kp_level_list.append([])
#     Xray_level_num_list=[]
#     # for i in range(len(date_list)):
#     #     Xray_level_num_list.append([])
#     # dict_final_data={}

# #######dst处理
#     # max_level_list=[]
#     # date_index=0
#     # for i in range(len(Dst_data)):
#     #     temp_level_list=[]
#     #     if date_index==8766:
#     #         break
#     #     cur_date=date_list[date_index]+' '
#     #     if cur_date in Dst_date[i]:
#     #         for j in range(24):
#     #             temp_level_list.append(int(Dst_level[i+j]))
#     #         max_level=max(temp_level_list)
#     #         max_level_list.append(max_level)
#     #         date_index=date_index+1
    
#     # dic={'time':date_list,'Max_Dst_level':max_level_list}
#     # df=pd.DataFrame(dic)
    
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Dst.csv", index=False)
# #######dst处理



# ######kp处理

#     # date_index=0
#     # for i in range(len(Kp_data)):
#     #     three_hour_level_list=[]
#     #     if date_index==8766:
#     #         break
#     #     cur_date=date_list[date_index]+' '
#     #     if cur_date in Kp_date[i]:
#     #         for j in range(0,24,3):
#     #             three_hour_level_list.append(Kp_level[i+j])
#     #         Kp_level_list.append(three_hour_level_list)
#     #         date_index=date_index+1


#     # Kp_level_list=np.array(Kp_level_list)

#     # # dic={'time':date_list,'Kp_level':Kp_level_list}
#     # dic={'time':date_list,'0-2':Kp_level_list[:,0],'3-5':Kp_level_list[:,1],'6-8':Kp_level_list[:,2],'9-11':Kp_level_list[:,3],'12-14':Kp_level_list[:,4],'15-17':Kp_level_list[:,5],'18-20':Kp_level_list[:,6],'21-23':Kp_level_list[:,7]}

#     # df=pd.DataFrame(dic)
    
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Kp.csv", index=False)


# #######kp处理


# #######xray处理

#     # Xray_event_54_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\X-ray_event\\New_x-ray_class\\new_X-ray_5.csv'
#     # Xray_event_32_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\X-ray_event\\New_x-ray_class\\new_X-ray_1.csv'
#     # Xray_event_C_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\X-ray_event\\X-ray_event_C.csv'

#     # Xray_file=pd.read_csv(Xray_event_54_path)
#     # # Xray_data = Xray_file.iloc[:, 1]  
#     # Xray_time = Xray_file.iloc[:, 0]   # 获取时间信息
#     # Xray_level = Xray_file.iloc[:, 1]

#     # # aaa=Xray_time[0]
#     # # bbb=datetime.strptime(aaa, "%Y-%m-%d %h-%m-%s")
#     # # ccc=bbb.strftime("%yyyy-%mm-%dd")
#     # Xray_time_new=[]
#     # for i in range(len(Xray_time)):
#     #     temp_time=Xray_time[i]
#     #     temp_time=temp_time[0:10]
#     #     Xray_time_new.append(temp_time)

#     # Xray_Class_sum_list=[]
#     # Xray_Class_sum=0
#     # date_index=0
#     # for date_index in date_list:
#     #     Xray_Class_sum=Xray_time_new.count(date_index)
#     #     Xray_Class_sum_list.append(Xray_Class_sum)

#     # for i in range(len(date_list)):
#     #     Xray_Class_sum_list.append(0)
#     # dic={'time':date_list,'X-ray(R5)':Xray_Class_sum_list}
#     # df=pd.DataFrame(dic)
    
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Xray_5.csv", index=False)

# #######xray处理

# #######gnss_time处理

#     # obs_date_list=[]
#     # for date_index in date_list:
#     #     dt=datetime.strptime(date_index,"%Y-%m-%d")
#     #     begin_date=dt-timedelta(days=7)
#     #     end_date=dt+timedelta(days=7)
#     #     begin_date=begin_date.strftime("%Y-%m-%d")
#     #     end_date=end_date.strftime("%Y-%m-%d")
#     #     obs_date_list.append(begin_date+'  to  '+end_date)
#     # dic={'date':date_list,'obs_time':obs_date_list}
#     # df=pd.DataFrame(dic)
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\obs_time.csv", index=False)


# #######gnss_time处理

#     date_list=get_date_list('1999-01-01', '2022-12-31',1,interval_type='day')
#     file_data = pd.read_csv('D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\total_event_v2.csv')
#     date=file_data.iloc[:,0]
#     dst=file_data.iloc[:,1]
#     kp1=file_data.iloc[:,3]
#     kp2=file_data.iloc[:,4]
#     kp3=file_data.iloc[:,5]
#     kp4=file_data.iloc[:,6]
#     kp5=file_data.iloc[:,7]
#     kp6=file_data.iloc[:,8]
#     kp7=file_data.iloc[:,9]
#     kp8=file_data.iloc[:,10]
#     Xray_1=file_data.iloc[:,12]
#     Xray_2=file_data.iloc[:,13]
#     Xray_3=file_data.iloc[:,14]
#     Xray_4=file_data.iloc[:,15]
#     Xray_5=file_data.iloc[:,16]
    
#     csv_list=[]

#     for i in range(len(date)):
#         Xray_1_=1 if Xray_1[i]>0 else 0
#         Xray_2_=2 if Xray_2[i]>0 else 0
#         Xray_3_=3 if Xray_3[i]>0 else 0
#         Xray_4_=4 if Xray_4[i]>0 else 0
#         Xray_5_=5 if Xray_5[i]>0 else 0
#         Xray_class=max(max(Xray_1_,Xray_2_),Xray_3_)
#         Xray_class=max(max(Xray_class,Xray_4_),Xray_5_)
#         Xray_class=str(Xray_class)
#         # day=date[i]
#         # if '2001-04-12 00:00:00'==day:
#         #     print("kokoko")
#         list_all=[]
#         dst_=dst[i]
#         # kkkp=str(dst[i])
#         # if math.isnan(int(dst[i])):
#         #     print('opppp')
#         #     dst_=0
#         list_all.append(str(dst[i]))
#         list_all.append(str(kp1[i]))
#         list_all.append(str(kp2[i]))
#         list_all.append(str(kp3[i]))
#         list_all.append(str(kp4[i]))
#         list_all.append(str(kp5[i]))
#         list_all.append(str(kp6[i]))
#         list_all.append(str(kp7[i]))
#         list_all.append(str(kp8[i]))
#         list_all.append(str(Xray_class))

#         kp_max=max(list_all[1:9])
#         xray_max=list_all[9]
#         # ppp=max(list_all[1:9])
       

#         query_level='1'
#         # if (query_level in list_all) and ('5' not in list_all) and ('4' not in list_all):
#         # if (query_level in list_all) and ('5' not in list_all) and ('4' not in list_all) and ('3' not in list_all) and ('2' not in list_all):
#         # if (query_level in list_all) and ('5' not in list_all) and ('4' not in list_all) and ('3' not in list_all) and ('2' not in list_all):
#         # if (query_level in list_all) and ('5' not in list_all) and ('4' not in list_all) and ('3' not in list_all)  and ('2' not in list_all):
#         if (query_level in list_all) and ('5' not in list_all) and ('4' not in list_all) and ('3' not in list_all) and ('2' not in list_all):
#             # csv_list.append(file_data.iloc[i,:])
#             temp_list=[date[i],file_data.iloc[i,1],kp_max,xray_max,file_data.iloc[i,18]]
#             # csv_list.append(date[i])
#             # csv_list.append(file_data.iloc[i,1])
#             # csv_list.append(kp_max)
#             # csv_list.append(xray_max)
#             # csv_list.append(file_data.iloc[i,18])
#             csv_list.append(temp_list)
#         else:
#             continue
            
#     df=pd.DataFrame(csv_list)
#     df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\level_1_list.csv", index=False)


# #######筛选

#     # Xray_event_X_path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\X-ray_event\\New_x-ray_class\\X-ray_event_M.csv'
#     # file_data=pd.read_csv(Xray_event_X_path)
#     # date=file_data.iloc[:,0]
#     # x_ray_data=file_data.iloc[:,1]
#     # new_date_list=[]
#     # new_level_list=[]
#     # for i in range(len(date)):
#     #     if x_ray_data[i]>5e-5:
#     #         new_date_list.append(date[i])
#     #         new_level_list.append(3)
#     #     elif x_ray_data[i]>1e-5:
#     #         new_date_list.append(date[i])
#     #         new_level_list.append(2)

#     # dic={'date':new_date_list,'X-ray_level':new_level_list}
#     # df=pd.DataFrame(dic)
#     # df.to_csv('D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\X-ray_event\\New_x-ray_class\\new_X-ray_3_2.csv',index=False)

# #######筛选




#     print('ok')
#     # csvpath="./SpaceWeatherData/origin_data/OMNI2_H0_MRG1HR_186271.csv"
#     # outpath="./SpaceWeatherData/processed_data/processed_OMNI2_H0_MRG1HR_186271.csv"
#     # process_F10_7_Dst_Ap(csvpath,outpath)

    
#     # path='D:\\恢复\\X-ray\\2016'
#     # outpath='D:\\恢复\\processed_X_ray_2016.csv'
#     # process_X_ray(path,outpath)

#     # data = pd.read_csv('./SpaceWeatherData/processed_data/processed_OMNI2_H0_MRG1HR_186271.csv')
#     # data_origin=data.iloc[:,0:len(data)]
#     # data_origin.plot(subplots=True,figsize=(30,10),
#     #         layout=(3,1),title='space weather features')
#     # plt.show()

#     # path='D:\恢复\CELIAS_SEM_15sec_avg'
#     # outpath='D:\\恢复\\processed_EUV_1999_2007.csv'
#     # process_EUV(path,outpath) 


#     # path='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\2016_2022\\2017-2022\\2022'
#     # outpath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\2016_2022\\2017-2022\\2022\\g16_xrs_1m_20220101_20221231.csv'
#     # process_X_ray_nc_2(path,outpath)


#     # data = pd.read_csv('C:\\solar_cezhan\\20150317\\code\\SpaceWeatherData\\processed_data\\total.csv')
#     # data_origin=data.iloc[:,0:len(data)]
#     # data_origin.index=data['TIME_AT_CENTER_OF_HOUR_yyyy-mm-ddThh:mm:ss.sssZ']
#     # data_origin.plot(subplots=True,figsize=(80,10),
#     #         layout=(5,1),title='space weather features')
#     # plt.show() 
    
#     # filepath='D:\\恢复\\TEC-CODE\\ion'
#     # outpath='D:\\恢复\\processed_tec-code_1999_2007.csv'
#     # process_TEC_single(filepath,outpath)

#     # TECfile='D:\\恢复\\TEC-CODE\\ion\\2014\\292\\final\\codg2920.14i'
#     # lltec = read_ionFile_1h(TECfile)
    
#     # path='D:\\恢复\\X-ray'
#     # outpath='D:\\恢复\\processed_X_ray_99_07.csv'
#     # process_X_ray(path,outpath)

#     # date1 = "2016-01-01"
#     # date2 = "2022-12-31"
#     # result = daysBetweenDates(date1, date2)
#     # print(f"{date1} 和 {date2} 之间相隔：{result}天。")
#     # print("OK")

#     # nc_PATH='D:\\恢复\\X-ray\\2021\\dn_xrsf-l2-avg1m_g16_y2021_v2-1-0.nc'
#     # nc_PATH='dn_xrsf-l2-avg1m_g16_y2021_v2-1-0.nc'
#     # nc_obj = Dataset(nc_PATH)
#     # #查看nc文件有些啥东东
#     # # print(nc_obj)
#     # print('---------------------------------------')
#     # print(nc_obj.variables.keys())
#     # for i in nc_obj.variables.keys():
#     #     print(i)
#     # print('---------------------------------------')

#     # # xray = (nc_obj.variables['xrsb_flux_observed'][:])
#     # xray = nc_obj['xrsb_flux_observed'][:].data  # 409个数字
#     # num=len(xray)
#     # temp=xray[0:num]
#     # temp=temp[0:44639]
#     # for i in range(0,44639):
#     #     if temp[i]<0 and i>1 :
#     #         temp[i]=temp[i-1]
#     # print('---------------------------------------')
#     # plt.plot(temp)
#     # # plt.ticklabel_format(style='sci', scilimits=(0,0))
#     # plt.ylim(1e-9,1e-2)
#     # plt.yscale('log')
#     # plt.show()



#     # path="D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Kp\\processed_Kp_3h_1999_2022_2.csv"
#     # data = pd.read_csv(path)
#     # feat = data.iloc[:, 1]  
#     # date = data.iloc[:, 0]   # 获取时间信息
#     # level=data.iloc[:, 2] 
#     # data_list=[]
#     # time_list=[]
#     # level_list=[]
#     # for i in range(len(data)):
#     #     if level[i]==1:
#     #         data_list.append(feat[i])
#     #         time_list.append(date[i])
#     #         level_list.append(level[i])
#     # dic={'time':time_list,'Dst':data_list,'Dst_level':level_list}
#     # df=pd.DataFrame(dic)
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Kp\\Kp_event\\Kp_event_小.csv", index=False)
#             # print(date[i])
#             # print('\n')

#     # list_=np.array([4,2,6,7,4,3])
#     # temp_list=[]
#     # for i in range(len(list_)):
#     #     temp=list_[i]
#     #     temp=temp if abs(temp)<6 else temp_list[i-1]
#     #     temp_list.append(temp)

#     # path="D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Kp\\processed_Kp_3h_1999_2022.csv"
#     # data = pd.read_csv(path)
#     # feat = data.iloc[:, 1]  
#     # date = data.iloc[:, 0]   # 获取时间信息
#     # temp_date_list=[]
#     # Kp_list=[]
#     # level_list=[]
#     # for i in range(len(data)):
#     #     temp_date=date[i]
#     #     # temp_date=temp_date[0:10]+' '+temp_date[11:19]
#     #     temp_date_list.append(temp_date)
#     #     temp_Kp=feat[i]
#     #     # temp_Kp=temp_Kp/10
#     #     # temp_Kp=temp_Kp if abs(temp_Kp)<11 else 0
#     #     # temp_Dst=temp_Dst if abs(temp_Dst)<500 else 0
#     #     # temp_Dst=temp_Dst/10
#     #     # temp_Dst=temp_Dst if abs(temp_Dst)<11 else 0

#     #     # round_temp_Dst=round(temp_Kp)
#     #     round_temp_Dst=temp_Kp
#     #     if round_temp_Dst<4.7:
#     #         level_list.append('0')
#     #     elif round_temp_Dst>=4.7 and round_temp_Dst<=5.3:
#     #         level_list.append('1')
#     #     elif round_temp_Dst>=5.7 and round_temp_Dst<=6.3:
#     #         level_list.append('2')
#     #     elif round_temp_Dst>=6.7 and round_temp_Dst<=7.3:
#     #         level_list.append('3')
#     #     elif round_temp_Dst>=7.7 and round_temp_Dst<=8.7:
#     #         level_list.append('4')
#     #     elif round_temp_Dst==9:
#     #         level_list.append('5')
#     #     elif round_temp_Dst==9.9:
#     #         temp_Kp=None
#     #         level_list.append('-1')
#     #     Kp_list.append(temp_Kp)
#     # dic={'time':temp_date_list,'3-H_Kp':Kp_list,'level':level_list}
#     # df=pd.DataFrame(dic)
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\Kp\\processed_Kp_3h_1999_2022_2.csv", index=False)

#     # a=data['time']
#     # data['time']=pd.to_datetime(data['time'],format="%Y-%m-%d %H:%M:%S")
#     # data.set_index('time',inplace=True)
#     # data['1-H_DST_nT'].plot()
    
#     # filepath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\2019\processed'
#     # #读取所有文件名
#     # Xray_filelist_unordered=get_obsFileAbPathList(filepath)
#     # for file in Xray_filelist_unordered:
#     #   newfilename='processed_'+file[-32:len(file)]
#     #   begin_num=0
#     #   flag=0 #g10和g11为1，其余为2
#     #   if 'g10' in file or 'g11' in file:
#     #         flag=1
#     #   elif 'g15' in file or 'g14' in file or 'g13' in file:
#     #         flag=2
#     #   Xray_list=[]
#     #   time_list=[]
#     #   with open(file,'r') as f:
#     #         line=f.readline()
#     #         while line:
#     #             if 'data:' in str(line):
#     #                 begin_num=begin_num+2
#     #                 break
#     #             begin_num=begin_num+1
#     #             line=f.readline()  
#     #   with open(file,'r') as f:
#     #         content=f.readlines()
#     #         content_=content[begin_num:len(content)]
#     #         for line in content_:
#     #             line=line.split(',')
#     #             Xray=line[2] if flag==1 else line[6]
#     #             time_list.append(line[0])
#     #             Xray_list.append(Xray)
#     #   dic={'time':time_list,'X-ray':Xray_list}
#     # #   df=pd.DataFrame(data=Xray_list,
#     # #                 columns=['X-ray'])
#     #   df=pd.DataFrame(dic)
#     #   df.to_csv(filepath+'\\'+newfilename,index=False)


#     # filepath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray'
#     # #读取所有文件名
#     # Xray_filelist_unordered=get_obsFileAbPathList(filepath)
#     # for file in Xray_filelist_unordered:
#     #     level_list=[]
#     #     if 'processed_' in file:
#     #         data=pd.read_csv(file)
#     #         xray = data.iloc[:, 1]  
#     #         time = data.iloc[:, 0]   # 获取时间信息
#     #         new_xray_list=[]
#     #         new_time_list=[]
#     #         for i in range(len(xray)):
#     #             new_xray=xray[i] if xray[i]>-9999 else None
#     #             if new_xray==None:
#     #                 new_xray_list.append(new_xray)
#     #                 new_time_list.append(time[i])
#     #                 level_list.append('0')
#     #                 continue
#     #             if new_xray<1e-7:
#     #                 level_list.append('1')
#     #             elif new_xray>=1e-7 and new_xray<1e-6:
#     #                 level_list.append('2')
#     #             elif new_xray>=1e-6 and new_xray<1e-5:
#     #                 level_list.append('3')
#     #             elif new_xray>=1e-5 and new_xray<1e-4:
#     #                 level_list.append('4')
#     #             elif new_xray>=1e-4:
#     #                 level_list.append('5')
#     #             new_xray_list.append(new_xray)
#     #             new_time_list.append(time[i])
#     #         dic={'time':new_time_list,'X-ray':new_xray_list,'X-ray_level':level_list}
#     #         df=pd.DataFrame(dic)
#     #         # temp=file[0:-42]
#     #         df.to_csv(file,index=False)
#     #         # plt.plot(new_xray_list)
#     #         # plt.ylim(1e-9,1e-2)
#     #         # plt.yscale('log')
#     #         # plt.show()
#     #         print('ok')

#     # filepath='D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray'
#     # #读取所有文件名
#     # Xray_filelist_unordered=get_obsFileAbPathList(filepath)
#     # time_list=[]
#     # data_list=[]
#     # level_list=[]
#     # for file in Xray_filelist_unordered:
#     #     if 'processed_' in file:
#     #         data=pd.read_csv(file)
#     #         feat=data.iloc[:,1]
#     #         time=data.iloc[:,0]
#     #         level=data.iloc[:,2]
#     #         for i in range(len(data)):
#     #             if level[i]>=5:
#     #                 data_list.append(feat[i])
#     #                 time_list.append(time[i])
#     #                 level_list.append(level[i])
#     # dic={'time':time_list,'X-ray':data_list,'X-ray_level':level_list}
#     # df=pd.DataFrame(dic)
#     # df.to_csv("D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\event\\X-ray_event_X.csv", index=False)
#     # print('ok')
#     # data=pd.read_csv('D:\\SpaceWeather_Project\\Data\\SpaceWeather_Data\\X-ray\\2010\\processed\\processed_g14_xrs_1m_20101101_20101130.csv')
#     # xray = data.iloc[:, 1]  
#     # time = data.iloc[:, 0]   # 获取时间信息
#     # new_xray_list=[]
#     # for i in range(len(xray)):
#     #     new_xray=xray[i] if xray[i]>-9999 else None
#     #     new_xray_list.append(new_xray)
#     # plt.plot(new_xray_list)
#     # plt.ylim(1e-9,1e-2)
#     # plt.yscale('log')
#     # plt.show()
#     # print('ok')

#     print('ok')
#     # Three subplots sharing both x/y axes
#     # x = np.linspace(0, 2 * np.pi, 400)
#     # y = np.sin(x ** 2)
#     # f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
#     # ax1.plot(x, y)
#     # ax1.set_title('Sharing both axes')
#     # ax2.scatter(x, y)
#     # ax3.scatter(x, 2 * y ** 2 - 1, color='r')
#     # # Fine-tune figure; make subplots close to each other and hide x ticks for
#     # # all but bottom plot.
#     # f.subplots_adjust(hspace=0)
#     # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

#     # plt.show()



