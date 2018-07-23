from agent2 import *
from mazeimage import *

if __name__ == '__main__':

    # init
    #ランダム行動確率ε
    epsilon = 0.1
    #学習率
    alpha = 0.2
    #割引率
    gamma = 0.9
    #csvファイル読み込み
    maze = numpy.loadtxt('./resources/sample.csv', delimiter = ',')


    #追加　2018/7/22
    #初期一座標を手動で入力
    x=input("start_x:"+"0~"+str(maze.shape[1]-1)+">>")
    y=input("start_y:"+"0~"+str(maze.shape[0]-1)+">>")
    start_x=int(x)
    start_y=int(y)
    start=[start_y, start_x]

    #ますの中で最大の報酬を取得
    max_reward=maze.max()
    #その座標を取得
    reward_indexes = numpy.argwhere(maze == max_reward)
    #ポケストップを通った数を数えるカウンター
    counter=0
    #終了条件：n回ポケストップを通ったらゴール
    limit=3
    #通ったポケストップの座標を記録
    poke_location=[start]
    #試行回数 20*20のマスでは100試行で130分かかる
    trial_max=300
    #通ったポケストップの座標（捨てる用）
    encount_indexes=[]

    agent = Agent(maze.shape, start, reward_indexes)
    maze_image = MazeImage(maze, 600, 600)

    trial = 0
    while True:
        if maze_image.show(agent) == 27:
            print('!!!escape!!!')
            break

        agent.act(maze, epsilon, alpha, gamma, counter, poke_location)
        maze_image.save_movie()

        #if agent.goal(maze.shape):

        #変更
        maze, reward_indexes, encount_indexes, counter, poke_location=agent.poke_count(reward_indexes, encount_indexes, counter, poke_location, maze, trial, trial_max)
        if agent.goal(counter, limit):
            #\033[32mとか\033[0mは文字色をつけたりするコンソール制御っぽい？
            print('' + '!!!goal!!!' + '')
            trial += 1
            print('next trial: %d' % trial)
            agent.reset(start, maze, encount_indexes, max_reward)
            encount_indexes=[]
            if trial < trial_max:
                poke_location=[start]
            counter=0

        if trial == trial_max:
            break

    maze_image.save_movie()
    #使うQ値の指定がややこしいから後回し
    #cv2.imwrite('shortest2.png', maze_image.shortest_path(agent.q))

    #初期座標と通ったポケストップの座標表示
    poke_location_xy=[]
    for i in range(len(poke_location)):
        xy=[poke_location[i][1], poke_location[i][0]]
        poke_location_xy.append(xy)
    print(poke_location_xy)
    cv2.destroyAllWindows()
