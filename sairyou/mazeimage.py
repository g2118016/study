import numpy
import cv2

class MazeImage:

    # constructor
    def __init__(self, maze, height, width):
        self.height, self.width = (height, width)
        self.maze_h, self.maze_w = maze.shape
        self.ystride = int(height / self.maze_h)
        self.xstride = int(width / self.maze_w)
        self.map_org = self.__create_image(maze)
        self.map_now = self.map_org
        #openCV3以降はcv_fource→cv2.VideoWriter_fourcc
        self.writer = cv2.VideoWriter('q_learning.mv4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (self.map_org.shape[1], self.map_org.shape[0]))


# public method
    def show(self, agent):
        self.map_now = self.map_org.copy()
        _y, _x = agent.state
        center = (int((_x + 0.5) * self.xstride), int((_y + 0.5) * self.ystride))
        cv2.circle(self.map_now, center, 5, (255, 255, 255), -1)
        cv2.imshow('', self.map_now)
        return cv2.waitKey(10)


    def save_movie(self):
        self.writer.write(self.map_now)


    def shortest_path(self, q):
        shortest_map = self.map_org.copy()
        q_arrow = numpy.vectorize(lambda x: {0: '<', 1: 'V', 2: '>', 3: '^'}.get(x))(q.argmax(axis = 0))

        for j in range(self.maze_h):
            for i in range(self.maze_w):
                pt = (int(self.xstride * (i + 0.35)), int(self.ystride * (j + 0.6)))
                cv2.putText(shortest_map, q_arrow[j, i], pt, cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255])

        return shortest_map


# private method
    def __create_image(self, maze):
        image = numpy.zeros((self.height, self.width, 3)).astype('uint8')

        for j in range(self.maze_h):
            for i in range(self.maze_w):
                tl = (self.xstride * i, self.ystride * j)
                br = (self.xstride * (i + 1) - 1, self.ystride * (j + 1) - 1)
                cv2.rectangle(image, tl, br, self.__color(maze[j, i]), -1)

        return image


    def __color(self, score):
        if score == 0.0:
            return [0, 0, 0]
        elif score == -1.0:
            return [0, 0, 127]
        else:
            return [127, 0, 0]
