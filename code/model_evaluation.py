from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

 
         
#load dataset
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

#load model
model = keras.models.load_model("./TEC_predict_model_48_48/model_saved")

# 使用sklearn调用衡量线性回归的MSE 、 RMSE、 MAE、r2
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
print("mean_squared_error:", mean_squared_error(y_test, y_predict))
print("rmse:", sqrt(mean_squared_error(y_test, y_predict)))
print("r2 score:", r2_score(y_test, y_predict))

#原生实现
# 衡量线性回归的MSE 、 RMSE、 MAE、r2
mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
rmse = sqrt(mse)
mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
r2 = 1-mse/ np.var(y_test) # 均方误差/方差
print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)
