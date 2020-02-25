import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd 
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import scipy.stats
import datetime


##データ読み込み
x_size =100
data = np.loadtxt('data.csv',delimiter=',')

##データをXとYに分割 #
data = data.T
X = data[:x_size].T
Y = data[x_size:].T

#スペクトルデータを正規化(zscore)
X = scipy.stats.zscore(X)

#学習データとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size=0.8)

#Yラベル(ペプチドアルファベット26個＋emptyの数字表現)をカテゴリ変数に変換
#大きさ　27*20
num_classes = 27
y_train = to_categorical(Y_train, num_classes = num_classes)
y_test = to_categorical(Y_test, num_classes = num_classes)
y_train=y_train.reshape((-1,20*27))
y_test=y_test.reshape((-1,20*27))

#隠れ層二層モデル構築

model = Sequential()

model.add(Dense(500, input_dim=100))
model.add(Activation("relu"))
model.add(Dropout(rate=0.2))

model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(rate=0.5))

model.add(Dense(20*num_classes))
model.add(Activation("softmax"))

#最適化関数 learning rate = 0.001
optimizer = Adam(lr=0.001)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#モデルの詳細表示
model.summary()

#重みの保存

w_dir = "./weights/"
if os.path.exists(w_dir) == False:os.mkdir(w_dir)
model_checkpoint = ModelCheckpoint(
    w_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)


# reduce_lr
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1
)

#tensor board
logging = TensorBoard(log_dir="log/")

#学習
hist = model.fit(
    X_train,
    y_train,
    verbose=1,
    epochs=50,
    batch_size = 32,
    validation_split=0.2,
    callbacks=[ reduce_lr, logging]
)

model_dir = './model/'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

dt_now = datetime.datetime.now()
now =dt_now.strftime('%Y_%m_%d_%H_%M_%S')
model.save(model_dir+now+'mmode_hdf5')


# accuracy
plt.subplot(1, 2, 1)
plt.plot(hist.history["acc"], label = "acc", marker = "o")
plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("accuracy")
#plt.title("")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha=0.2)

# loss
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label = "loss", marker = "o")
plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("loss")
#plt.title("")
plt.legend(loc="best")
plt.grid(color = 'gray', alpha = 0.2)

plt.savefig(now+"train_curve.png")
score = model.evaluate(X_test, y_test, verbose=1)
print("evaluate loss: {0[0]}".format(score))
print("evaluate acc: {0[1]}".format(score))