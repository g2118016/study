import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.training import extensions

# データセットの準備
indata = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
labels = np.array([ [0],  [1],  [1],  [0] ], dtype=np.float32)

dataset = chainer.datasets.TupleDataset(indata, labels)
train_iter = chainer.iterators.SerialIterator(dataset, 4) # 訓練用 batch size = 4
test_iter = chainer.iterators.SerialIterator(dataset, 4, repeat=False, shuffle=False)
                                                          # 評価用 batch size = 4

#L.Linear(in_size, out_size, wscale=1, bias=0, nobias=False,
#         initialW=None, initial_bias=None)
class Xor(chainer.Chain):  #chainer.Chainを継承したクラスで全体の関数を表現
    def __init__(self): # constructor
        super(Xor, self).__init__( #chainer.Chainでlinkをまとめる
            l1 = L.Linear(2, 10),
            l2 = L.Linear(10, 1)
        )

    def __call__(self, x): #forward propagation definition
        h1 = F.relu(self.l1(x))
        return self.l2(h1)


my_xor = Xor() # 訓練したい関数を示すモデル
accfun = lambda x, t: F.sum(1 - abs(x-t))/x.size #ないと動かないのでとりあえず用意
model = L.Classifier(my_xor, lossfun=F.mean_squared_error, accfun=accfun)
                                     # datasetのデータを入力するとlossを返すモデル

optimizer = chainer.optimizers.Adam()
optimizer.setup(model) #lossを返すモデルをセット
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# 訓練の準備
updater = chainer.training.StandardUpdater(train_iter, optimizer)
trainer = chainer.training.Trainer(updater, (1000, 'epoch'), out="test_result")
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar())
print('start')
trainer.run()
print('done')
