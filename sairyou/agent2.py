import numpy
import cv2
import random
import copy
import itertools

class Agent:

    # constructor
    def __init__(self, maze_shape, start, reward_indexes):
        #初期位置座標
        self.state = numpy.array(start)
        #各マスにおける可能な行動を生成 十字キーに0~3を振り分ける
        self.actions = self.__create_actions(maze_shape)
        #十字キーと実際の動きを対応させる
        self.moves = {0: numpy.array([0, -1]), 1: numpy.array([1, 0]), 2: numpy.array([0, 1]), 3: numpy.array([-1, 0])}
        #4*ステージの大きさのQ値格納用のリスト？
        #各マスの各行動（上，下，左，右）の行動ごとの価値を入れる


        self.default_q = numpy.zeros((4, ) + maze_shape)
        self.one_q = {}
        for i in reward_indexes:
            self.one_q[tuple(i)] = numpy.zeros((4, ) + maze_shape)

        self.two_q = {}
        temp_list=[]
        for i in reward_indexes:
            temp_list.append(tuple(i))
        collection = list(itertools.permutations(temp_list, 2))
        for j in collection:
            self.two_q[j] = numpy.zeros((4, ) + maze_shape)



# public method
    def act(self, maze, epsilon, alpha, gamma, counter, poke_location):
        #行動を選択（0~3）
        act_index = self.__select_action(epsilon, counter, poke_location)
        #行動に対応する座標を取得
        move = self.moves.get(act_index)
        #移動した先の報酬を取得
        reward = maze[tuple(self.state + move)]
        #Q値の更新
        self.update_q(act_index, move, reward, alpha, gamma, counter, poke_location)
        self.state += move


    def update_q(self, act_index, move, reward, alpha, gamma, counter, poke_location):
        y, x = self.state
        if counter==0:
            _q = self.default_q[act_index, y, x]
            self.default_q[act_index, y, x] = _q + alpha * (reward + gamma * self.__get_max_default_q(move) - _q)
        elif counter==1:
            _q = self.one_q[tuple(poke_location[1])][act_index, y, x]
            self.one_q[tuple(poke_location[1])][act_index, y, x] = _q + alpha * (reward + gamma * self.__get_max_one_q(move, tuple(poke_location[1])) - _q)
        elif counter==2:
            poke_2=(tuple(poke_location[1]), tuple(poke_location[2]))
            for i in self.two_q.keys():
                if poke_2 == i and poke_2[::-1] == i:
                    poke_2=i
            _q = self.two_q[poke_2][act_index, y, x]
            self.two_q[poke_2][act_index, y, x] = _q + alpha * (reward + gamma * self.__get_max_two_q(move, poke_2) - _q)


    """
    def goal(self, maze_shape):
        #右下についたらゴールと判定？
        return numpy.array_equal(self.state, numpy.array(maze_shape) - 1)
    """

    #変更　2018/7/22
    def poke_count(self, reward_indexes, encount_indexes, counter, poke_location, maze, trial, trial_max):
        #現在地がポケストップか判定
        for i in range(len(reward_indexes)):
            if numpy.array_equal(self.state, reward_indexes[i]) and list(self.state) not in encount_indexes:
                #コピー
                copy_zahyou=copy.deepcopy(reward_indexes[i]).tolist()

                #後で使うのでポケストップの場所を保存
                poke_location.append(copy_zahyou)

                counter+=1

                #1回通ったポケストップをカウントしないように削除
                """
                reward_indexes=reward_indexes.tolist()
                reward_indexes.remove(copy_zahyou)
                reward_indexes=numpy.array(reward_indexes)
                """
                #通ったポケストップを記録
                encount_indexes.append(copy_zahyou)

                #print("poke_location:", len(poke_location))
                #print("len(reward_indexes)", len(reward_indexes))
                #reward_indexes=numpy.delete(reward_indexes, numpy.where(reward_indexes==copy_zahyou))
                #通ったポケストップの報酬をただの道と同じ0にする
                maze[tuple(copy_zahyou)]=0
                break
        #print("counter:", counter)
        return maze, reward_indexes, encount_indexes, counter, poke_location


    def goal(self, counter, limit):
        if counter==limit:
            return True
        else:
            return False


    def reset(self, start, maze, encount_indexes, max_reward):
        #初期位置へ戻す
        self.state = numpy.array(start)
        for i in encount_indexes:
            maze[tuple(i)]=max_reward


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


    def __select_action(self, epsilon, counter, poke_location):
        #現在の位置を取得
        y, x = self.state
        #現在のマスからとれる行動（の選択肢）を取得
        action = copy.deepcopy(self.actions[y, x])
        #0~1未満の数字を生成，εより大きければQ値によって行動を選択
        if numpy.random.rand() > epsilon:
            mode = '!!!greedy!!!'
            act_index = self.__select_greedy_action(action, counter, poke_location)
        else:
            #小さければランダムな行動をとる
            mode = '!!!random!!!'
            act_index = self.__select_random_action(action)

        #print(u'%s  state: (%d, %d), action: %d' % (mode, y, x, act_index))

        return act_index


    def __select_greedy_action(self, action, counter, poke_location):
        y, x = self.state
        #価値が最大になる行動の価値（Q値）を取ってくる
        if counter==0:
            _max = self.default_q[action, y, x].max()
            #そのインデックスを取得
            _indexes = list(numpy.argwhere(self.default_q[action, y, x] == _max))
        elif counter==1:
            _max = self.one_q[tuple(poke_location[1])][action, y, x].max()
            #そのインデックスを取得
            _indexes = list(numpy.argwhere(self.one_q[tuple(poke_location[1])][action, y, x] == _max))
        elif counter==2:
            poke_2=(tuple(poke_location[1]), tuple(poke_location[2]))
            for i in self.two_q.keys():
                if poke_2 == i and poke_2[::-1] == i:
                    poke_2=i
            #print(poke_2)
            _max = self.two_q[poke_2][action, y, x].max()
            #そのインデックスを取得
            _indexes = list(numpy.argwhere(self.two_q[poke_2][action, y, x] == _max))
        #価値が最大の行動が複数ある場合はその中からランダムに行動を選ぶ
        random.shuffle(_indexes)
        #[0]を追加．_indexes[0]が[数字]になってたので
        return action[_indexes[0][0]]


    def __select_random_action(self, action):
        random.shuffle(action)
        return action[0]


    def __get_max_default_q(self, move):
        #移動先のマスの座標
        y, x = self.state + move
        #移動先で取れる行動の選択肢
        move = self.actions[y, x]
        #移動先のQ値の最大値を返す
        return self.default_q[move, y, x].max()


    def __get_max_one_q(self, move, poke_location):
        #移動先のマスの座標
        y, x = self.state + move
        #移動先で取れる行動の選択肢
        move = self.actions[y, x]
        #移動先のQ値の最大値を返す
        return self.one_q[poke_location][move, y, x].max()


    def __get_max_two_q(self, move, poke_location):
        #移動先のマスの座標
        y, x = self.state + move
        #移動先で取れる行動の選択肢
        move = self.actions[y, x]
        #移動先のQ値の最大値を返す
        return self.two_q[poke_location][move, y, x].max()
