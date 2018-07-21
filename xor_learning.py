#coding:utf-8

import chainer.links as L
import chainer.functions as F
from chainer import Chain
import chainer
from chainer import optimizers, cuda, serializers
import argparse
import numpy as np
import matplotlib.pyplot as plt

class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(2, 5),
            l2 = L.Linear(5, 1) )

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

    def predict(self, x):
        h1 = F.sigmoid(self.l1(x))
        return model.l2(h1)

#__call__よく分かんないから関数で代わり？
def forward(x, t):
    h1 = F.sigmoid(model.l1(x))
    return model.l2(h1)

"""
#引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=50000,          help='number of epochs to learn')

args = parser.parse_args()
n_epoch     = args.epoch        # エポック数(パラメータ更新回数)
"""

#学習データ
source = [[0, 0], [1, 0], [0, 1], [1, 1]]
target = [[0], [1], [1], [0]]
dataset = {}
dataset['source'] = np.array(source, dtype=np.float32)
dataset['target'] = np.array(target, dtype=np.float32)

model = MyChain()
optimizer = optimizers.Adam()  # 最適化手法をSGDに指定
optimizer.setup(model)

# Learning loop
loss_val = 100
epoch = 0

#グラフ用
graph_x=[]
graph_y=[]

while loss_val > 1e-5:

    # training
    x = chainer.Variable(np.asarray(dataset['source'])) #source
    t = chainer.Variable(np.asarray(dataset['target'])) #target

    model.zerograds()       # 勾配をゼロ初期化
    y    = forward(x, t)    # 順伝搬

    loss = F.mean_squared_error(y, t) #平均二乗誤差

    loss.backward()              # 誤差逆伝播
    optimizer.update()           # 最適化

    #誤差と正解率を計算
    loss_val = loss.data
    #yに格納
    graph_y.append(loss_val)

    # 途中結果を表示
    if epoch % 1000 == 0:

        print('epoch:', epoch)
        print('x:\n', x.data)
        print('t:\n', t.data)
        print('y:\n', y.data)

        print('train mean loss={}'.format(loss_val)) # 訓練誤差, 正解率
        print(' - - - - - - - - - ')

    epoch += 1

    # n_epoch以上になると終了
    if epoch >= 10000:
        break


print("total:", epoch)

graph_x=np.arange(0, epoch, 1)
graph_y=np.array(graph_y)

# 横軸の変数。縦軸の変数。
plt.plot(graph_x, graph_y)

plt.xlabel("epochs")
plt.ylabel("Training error")
# 描画実行
plt.show()


#modelとoptimizerを保存
print('save the model')
serializers.save_npz('xor_mlp.model', model)
print('save the optimizer')
serializers.save_npz('xor_mlp.state', optimizer)


#serializers.load_npz('xor_mlp.model', model)
x1=int(input())
x2=int(input())
x=[[x1, x2]]
x=chainer.Variable(np.asarray(x))
y = model.predict(x)
print(y.data)
