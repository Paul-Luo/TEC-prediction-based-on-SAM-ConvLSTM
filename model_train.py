import tensorflow as tf
import os
from keras import Model, Sequential
# from keras.layers import ConvLSTM2D, LSTM, Conv2D, LeakyReLU, BatchNormalization, Activation,Conv3D
from convLSTM_model_TEC_with_SWdata import create_TEC_dataset,TimeSeries
import numpy as np
import pandas as pd
import keras
from tensorflow.data import Dataset
import SaConvLSTM
from SaConvLSTM import SaConvLSTM2D

class TECModel(Model):

  def __init__(self):
    super(TECModel, self).__init__()
    self.conv2d_lstm_1 = SaConvLSTM2D(filters=32, kernel_size=(7, 7),
                    padding='same',activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True)
    self.conv2d_lstm_2 = SaConvLSTM2D(filters=32, kernel_size=(5, 5),
                    padding='same',activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True)
    self.conv2d_lstm_3 = SaConvLSTM2D(filters=32, kernel_size=(3, 3),
                    padding='same',activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True)
    self.conv2d_lstm_4 = SaConvLSTM2D(filters=32, kernel_size=(1, 1),
                    padding='same',activation=LeakyReLU(alpha=0.01), 
                    return_sequences=True)
    self.conv3d = Conv3D(filters=1, kernel_size=(3,3,3),
                padding='same', data_format='channels_last')

    self.space_process = Sequential([
        self.conv2d_lstm_1, BatchNormalization(),
        self.conv2d_lstm_2, BatchNormalization(),
        self.conv2d_lstm_3, BatchNormalization(),
        self.conv2d_lstm_4, BatchNormalization(),
        self.conv3d, BatchNormalization()
    ])

    self.lstm = LSTM(units=48)

    self.combine = Sequential([
        Conv3D(filters=1, kernel_size=(1, 1, 1)),
        BatchNormalization(),
        Activation('sigmoid')
    ])

  def call(self, inputs):
    feature, tec = inputs
    # tec shape: [batch_size,6,73,71,1]
    # feature shape: [batch_size,6,5]
    fea_result = self.lstm(feature) 
    space_result = self.space_process(tec) 
    fea_result = tf.reshape(fea_result, [-1,48,1,1,1]) 
    fea_result = tf.repeat(fea_result, repeats=[space_result.shape[2]], axis=2)  
    fea_result = tf.repeat(fea_result, repeats=[space_result.shape[3]], axis=3)  
    return self.combine(tf.concat([fea_result, space_result], axis=4))   

  def predict(self, inputs):
    feature, tec, y = inputs
    return self([feature, tec])




########################################################################################################################################


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 调用GPU加速
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# TEC_data_list=np.load('D:\\tempdata\\TEC_20230422_20230425.npy')
TEC_data_list=np.load('tec_data_2010_2022_list.npy')
TEC_data_list=TEC_data_list.astype('float16')


SW_filepath = '1999_2022_f107_ssn_xray_ap_dst.csv'
data = pd.read_csv(SW_filepath)
feat=data.iloc[:,1:5]
# 求训练集的每个特征列的均值和标准差
feat_mean = feat.mean(axis=0)
feat_std = feat.std(axis=0)
# 对整个数据集计算标准差
feat = (feat - feat_mean) / feat_std
feat = feat.astype('float16')

# 构造训练集 (history_size大小必须与target_size一致)
X_train_,X_train_TEC, y_train_ = TimeSeries(dataset=feat, start_index=0, history_size=48, end_index=43000,
                            step=1, target_size=48, point_time=False, true=TEC_data_list)

# 构造验证集
X_val_,X_val_TEC, y_val_ = TimeSeries(dataset=feat, start_index=43000, history_size=48, end_index=46000,
                        step=1, target_size=48, point_time=False, true=TEC_data_list)


X_train_TEC=np.expand_dims(X_train_TEC, axis=-1)
X_val_TEC=np.expand_dims(X_val_TEC, axis=-1)


batch_size = 24
learning_rate_ = 0.00001
epochs = 30

feature_data = X_train_
tec_data = X_train_TEC
y_data = y_train_
train_dataset = Dataset.from_tensor_slices((feature_data, tec_data, y_data)).batch(batch_size) # generate  training dataset
test_dataset = Dataset.from_tensor_slices((X_val_, X_val_TEC, y_val_)).batch(batch_size) # generate testing dataset

model = TECModel() # create model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_), loss=tf.keras.losses.MeanSquaredError())

print("Training begin!")
for e in range(epochs):
    print(f"\nEpochs: {e+1}\n________________________")
    total_train_batchs = len(train_dataset)
    for num, elem in enumerate(train_dataset.shuffle(10)):
        feature, tec, y = elem
        with tf.GradientTape() as tape:
            y_pred = model([feature, tec])  
            loss = model.compiled_loss(y, y_pred)
        weights = model.space_process.trainable_weights + model.lstm.trainable_weights + model.combine.trainable_weights
        grads = tape.gradient(loss, weights)
        model.optimizer.apply_gradients(zip(grads, weights))

    # print loss every 10 batchs
        # if (num+1)%1000 == 0:
        if (num+1)%1000 == 0:
            print(f"loss({num+1}/{total_train_batchs}): {loss:>7f}")
    
    average_validation_loss = 0
    total_test_batchs = len(test_dataset)
    for num, elem in enumerate(test_dataset):
        feature, tec, y = elem
        y_pred = model([feature, tec])
        y_pred = tf.reshape(y_pred, [batch_size,48,71,73])
        average_validation_loss += model.compiled_loss(y, y_pred)

        # print average validation loss
        if (num+1) == total_test_batchs:
            print(f"Validation loss: {average_validation_loss:>7f}")

print("Training end!")
model.save('./TEC_predict_model_48_48/model_saved')
print("ok")






