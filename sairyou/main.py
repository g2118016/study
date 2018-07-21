from agent import *
from mazeimage import *

if __name__ == '__main__':

    # init
    epsilon = 0.1
    alpha = 0.2
    gamma = 0.9
    maze = numpy.loadtxt('./resources/maze.csv', delimiter = ',')
    agent = Agent(maze.shape)
    maze_image = MazeImage(maze, 300, 300)

    trial = 0
    while True:
        if maze_image.show(agent) == 27:
            print('!!!escape!!!')
            break

        agent.act(maze, epsilon, alpha, gamma)
        maze_image.save_movie()

        if agent.goal(maze.shape):
            print('\033[32m' + '!!!goal!!!' + '\033[0m')
            trial += 1
            print('next trial: %d' % trial)
            agent.reset()

        if trial == 300:
            break

    maze_image.save_movie()
    cv2.imwrite('shortest.png', maze_image.shortest_path(agent.q))
    cv2.destroyAllWindows()
