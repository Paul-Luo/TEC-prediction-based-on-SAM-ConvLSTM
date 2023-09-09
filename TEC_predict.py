import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 调用GPU加速
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # 构造测试集
    #从文件里读取TEC二维数组，存入list
    # TEC_data_list=create_TEC_dataset('D:\\TEC\\new_TEC_data\\')
    # np.save('D:\\TEC\\tec_data_2010_2022_list.npy',TEC_data_list)
    TEC_data_list=np.load('D:\\TEC\\tec_data_1999_2022_list.npy')
    TEC_data_list=TEC_data_list.astype('float16')
#（1）读取空间天气参数数据集
    SW_filepath = './1999_2022_f107_ssn_xray_ap_dst.csv'
    data = pd.read_csv(SW_filepath)
 
    # #（2）特征选择
    # # 选择从第1列开始往后的所有行的数据(第一列为时间序列)
    feat = data.iloc[0:, 1:6]  
    date = data.iloc[0:, 0]   # 获取时间序列

    # 求训练集的每个特征列的均值和标准差
    feat_mean = feat.mean(axis=0)
    feat_std = feat.std(axis=0)

    # 对整个数据集计算标准差
    feat = (feat - feat_mean) / feat_std
    train_num = 1  
    val_num = 10000  
    #（6）划分数据集
    history_size = 48  
    target_size =  24  
    step = 1  # 步长为1取所有的行
    x_test,X_test_TEC, y_test =  TimeSeries(dataset=feat, start_index=130000, history_size=history_size, end_index=140000,
                        step=step, target_size=target_size, point_time=False, true=TEC_data_list)

    model = keras.models.load_model("./TEC_predict_model_48_48/model_saved")


    X_test_choosed = x_test[1]
    X_test_choosed=np.expand_dims(X_test_choosed, axis=0)
    X_test_TEC_choosed=X_test_TEC[1]
    X_test_TEC_choosed=np.expand_dims(X_test_TEC_choosed, axis=-1)
    X_test_TEC_choosed=np.expand_dims(X_test_TEC_choosed, axis=0)


    new_prediction_1 = model([X_test_choosed, X_test_TEC_choosed])
    new_prediction_1 = np.squeeze(new_prediction_1, axis=0)
    new_prediction_temp = new_prediction_1
    SAM_new_prediction_1 = new_prediction_1 * 150


    nlats, nlons = 71, 73
    lats = np.linspace(-90, 90, nlats)
    lons = np.linspace(-180, 180, nlons)
    lons, lats = np.meshgrid(lons, lats)
    level=range(51)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=0))
    tick_proj = ccrs.PlateCarree()
    ax.set_xticks(np.arange(-180, 180+60, 60), crs=tick_proj)
    ax.set_xticks(np.arange(-180, 180+30, 30), minor=True, crs=tick_proj)
    ax.set_yticks(np.arange(-90, 90+30, 30), crs=tick_proj)
    ax.set_yticks(np.arange(-90, 90+15, 15), minor=True, crs=tick_proj)
    ax.set_global()
    ax.coastlines()
    cp = plt.contourf(lons, lats, (SAM_new_prediction_1).reshape((71,73)),level, transform=ccrs.PlateCarree(), cmap='jet')

    position = fig.add_axes([0.118, 0.05, 0.79, 0.014 ])#位置[左,下,宽,高]
    cc=plt.colorbar(cp, cax=position, orientation='horizontal', label= 'TEC(TECU)', ticks=[0,10,20,30,40,50])

    plt.show()









