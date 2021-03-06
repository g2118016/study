import numpy
import cv2
import random
import copy

class Agent:

    # constructor
    def __init__(self, maze_shape, start):
        #初期位置座標
        self.state = numpy.array(start)
        #各マスにおける可能な行動を生成 十字キーに0~3を振り分ける
        self.actions = self.__create_actions(maze_shape)
        #十字キーと実際の動きを対応させる
        self.moves = {0: numpy.array([0, -1]), 1: numpy.array([1, 0]), 2: numpy.array([0, 1]), 3: numpy.array([-1, 0])}
        #4*ステージの大きさのQ値格納用のリスト？
        #各マスの各行動（上，下，左，右）の行動ごとの価値を入れる
        self.q = numpy.zeros((4, ) + maze_shape)


# public method
    def act(self, maze, epsilon, alpha, gamma):
        #行動を選択（0~3）
        act_index = self.__select_action(epsilon)
        #行動に対応する座標を取得
        move = self.moves.get(act_index)
        #移動した先の報酬を取得
        reward = maze[tuple(self.state + move)]
        #Q値の更新
        self.update_q(act_index, move, reward, alpha, gamma)
        self.state += move


    def update_q(self, act_index, move, reward, alpha, gamma):
        y, x = self.state
        _q = self.q[act_index, y, x]
        self.q[act_index, y, x] = _q + alpha * (reward + gamma * self.__get_max_q(move) - _q)


    """
    def goal(self, maze_shape):
        #右下についたらゴールと判定？
        return numpy.array_equal(self.state, numpy.array(maze_shape) - 1)
    """

    #変更　2018/7/22
    def goal(self, reward_indexes, counter, limit, poke_location, maze):
        #現在地がポケストップか判定
        for i in range(len(reward_indexes)):
            if numpy.allclose(self.state, reward_indexes[i]):
                counter+=1
                #後で使うのでポケストップの場所を保存
                copy_index=copy.deepcopy(reward_indexes[i]).tolist()
                poke_location.append(copy_index)
                #1回通ったポケストップをカウントしないように削除
                numpy.delete(reward_indexes, copy_index)
                #通ったポケストップの報酬をただの道と同じ0にする
                maze[tuple(copy_index)]=0
                break
        if counter==limit:
            return True
        else:
            return False

    def reset(self):
        #初期位置へ戻す
        self.state = numpy.array([0, 0])


# private method
    def __create_actions(self, maze_shape):
        actions = []
        maze_h, maze_w = maze_shape
        for j in range(maze_h):
            actions.append([])
            for i in range(maze_w):
                action = [0, 1, 2, 3]
                self.__remove_actions(action, maze_h, maze_w, j, i)
                actions[j].append(action)

        return numpy.array(actions)


    def __remove_actions(self, action, maze_h, maze_w, j, i):
        if i == 0:
            action.remove(0)
        if i == maze_w - 1:
            action.remove(2)
        if j == 0:
            action.remove(3)
        if j == maze_h - 1:
            action.remove(1)


    def __select_action(self, epsilon):
        #現在の位置を取得
        y, x = self.state
        #現在のマスからとれる行動（の選択肢）を取得
        action = copy.deepcopy(self.actions[y, x])
        #0~1未満の数字を生成，εより大きければQ値によって行動を選択
        if numpy.random.rand() > epsilon:
            mode = '!!!greedy!!!'
            act_index = self.__select_greedy_action(action)
        else:
            #小さければランダムな行動をとる
            mode = '!!!random!!!'
            act_index = self.__select_random_action(action)

        print(u'%s  state: (%d, %d), action: %d' % (mode, y, x, act_index))

        return act_index


    def __select_greedy_action(self, action):
        y, x = self.state
        #価値が最大になる行動の価値（Q値）を取ってくる
        _max = self.q[action, y, x].max()
        #そのインデックスを取得
        _indexes = list(numpy.argwhere(self.q[action, y, x] == _max))
        #価値が最大の行動が複数ある場合はその中からランダムに行動を選ぶ
        random.shuffle(_indexes)
        #[0]を追加．_indexes[0]が[数字]になってたので
        return action[_indexes[0][0]]


    def __select_random_action(self, action):
        random.shuffle(action)
        return action[0]


    def __get_max_q(self, move):
        #移動先のマスの座標
        y, x = self.state + move
        #移動先で取れる行動の選択肢
        move = self.actions[y, x]
        #移動先のQ値の最大値を返す
        return self.q[move, y, x].max()
