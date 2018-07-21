from agent import *
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
    maze = numpy.loadtxt('./resources/maze.csv', delimiter = ',')


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
    limit=10
    #通ったポケストップの座標を記録
    poke_location=[start]


    agent = Agent(maze.shape, start)
    maze_image = MazeImage(maze, 600, 600)

    trial = 0
    while True:
        if maze_image.show(agent) == 27:
            print('!!!escape!!!')
            break

        agent.act(maze, epsilon, alpha, gamma)
        maze_image.save_movie()

        #if agent.goal(maze.shape):

        #変更
        if agent.goal(reward_indexes, counter, limit, poke_location, maze):
            #\033[32mとか\033[0mは文字色をつけたりするコンソール制御っぽい？
            print('' + '!!!goal!!!' + '')
            trial += 1
            print('next trial: %d' % trial)
            agent.reset()

        if trial == 300:
            break

    maze_image.save_movie()
    cv2.imwrite('shortest.png', maze_image.shortest_path(agent.q))

    #初期座標と通ったポケストップの座標表示
    print(poke_location)
    cv2.destroyAllWindows()
