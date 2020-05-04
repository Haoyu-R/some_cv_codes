from math import *
import random



landmarks = [[20, 20],[80,80], [20, 80], [80,20]]



world_size = 100

class Robot:
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2 * pi
        self.forward_noise = 0
        self.turn_noise = 0
        self.sense_noise = 0

    def __str__(self):
        return "[x={}, y={}, heading={}] \n".format(self.x, self.y, self.orientation)

    __repr__ = __str__

    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError("X out of bound")
        if new_y < 0 or new_y >= world_size:
            raise ValueError("Y out of bound")
        if new_orientation < 0 or new_orientation > 2*pi:
            raise ValueError("Orientation out of bound")
        self.x = new_x
        self.y = new_y
        self.orientation = new_orientation

    def set_noise(self, f_noise, t_noise, s_noise):
        self.forward_noise = f_noise
        self.turn_noise = t_noise
        self.sense_noise = s_noise

    def move(self, turn, forward):
        if forward < 0:
            raise ValueError("Robot cant move backwards")

        orientation = self.orientation + float(turn) + random.gauss(0, self.turn_noise)
        orientation %= 2*pi

        dist = float(forward) + random.gauss(0, self.forward_noise)
        x = self.x + cos(orientation)*dist
        y = self.y + sin(orientation)*dist
        x%=world_size
        y%=world_size
        self.x = x
        self.y = y
        self.orientation = orientation

        res = Robot()
        res.set(self.x, self.y, self.orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def Gaussian(self, mu, sigma, x):
        return (1.0/sqrt(2*pi*(sigma**2))) * exp(-0.5 * ((mu-x)/sigma)**2)

    def measurement_prob(self, measurement):
        prob = 1
        for i in range(len(landmarks)):
            dist = sqrt(pow(self.x-landmarks[i][0], 2)+pow(self.y-landmarks[i][1], 2))
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob

    def sense(self):
        z = []
        for i in range(len(landmarks)):
            dist = sqrt(pow(self.x-landmarks[i][0], 2)+pow(self.y-landmarks[i][1], 2))+random.gauss(0, self.sense_noise)
            z.append(dist)
        return z

T = 10
N = 1000
p = []

for i in range(N):
    x = Robot()
    x.set_noise(0.05, 0.05, 5)
    p.append(x)

for i in range(T):
    my_robot = Robot()
    # my_robot.set_noise(5, 0.1, 5)
    # my_robot.set(30, 50, pi/2)
    # my_robot.move(-pi/2, 15)
    # print(my_robot.sense())
    # my_robot.move(-pi/2, 10)
    # print(my_robot.sense())
    # print(my_robot)
    z = my_robot.sense()

    p2 = []

    for i in range(N):
        p2.append(p[i].move(0.1, 5))
    p = p2

    w = []
    for i in range(N):
        w.append(p[i].measurement_prob(z))

    p3 = []
    index = int(random.random()*N)
    beta = 0
    mw = max(w)

    for i in range(N):
        beta += random.random() * 2*mw;
        while beta > w[index]:
            beta -= w[index]
            index = (index+1) % N
        p3.append(p[index])

    p = p3

print(p)