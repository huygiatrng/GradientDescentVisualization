import math, copy
import pygame
import numpy as np

# initialize all imported pygame modules
pygame.init()
# Set dimension
screen = pygame.display.set_mode((1100, 600))
# Set Caption
pygame.display.set_caption("Gradient Descent Visualization")
clock = pygame.time.Clock()

# Colors set
BACKGROUND = ((0, 179, 179))
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND_PANEL = (204, 230, 255)
ALGORITHM_BACKGROUND = (255, 255, 102)

RED = (204, 0, 0)
YELLOW = (255, 255, 153)
BLUE = (77, 77, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
SKY = (0, 255, 255)
ORANGE = (255, 125, 25)
GRAPE = (100, 25, 125)
GRASS = (55, 155, 65)
PINK = (204, 0, 102)

COLORS = [RED, YELLOW, BLUE, GREEN, PURPLE, SKY, ORANGE, GRAPE, GRASS, PINK]

k = 1
error_number = 0
points = []
clusters = []
labels = []


def distanceOfTwoPoints(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Render text
def render_text(string, size):
    font = pygame.font.SysFont('dejavuserif', size)
    return font.render(string, True, WHITE)


def render_Ktext(string, size):
    font = pygame.font.SysFont('comicsansms', size)
    return font.render(string, True, BLUE)


def render_Errortext(string, size):
    font = pygame.font.SysFont('comicsansms', size)
    return font.render(string, True, RED)


def drawPanel():
    pygame.draw.rect(screen, BLACK, (50, 50, 800, 500))
    pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 790, 490))


def drawInterface():
    pygame.draw.rect(screen, BLACK, (860, 190, 150, 50))
    pygame.draw.rect(screen, BLACK, (860, 260, 200, 50))
    screen.blit(render_text("ALGORITHM", 28), (873, 260))
    screen.blit(render_text("RUN", 40), (890, 190))
    if 55 < mouse_x < 845 and 55 < mouse_y < 545:
        screen.blit(render_Ktext("(" + str(mouse_x - 55) + ", " + str(mouse_y - 55) + ")", 11), (mouse_x + 10, mouse_y))


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, cost_function, gradient_function, i, J_history, p_history):
    i += 1
    b = b_in
    w = w_in
    dj_dw, dj_db = gradient_function(x, y, w, b)
    b = b - alpha * dj_db
    w = w - alpha * dj_dw

    J_history.append(cost_function(x, y, w, b))
    p_history.append([w, b])
    print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
          f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
          f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history, i  # return w and J,w history for graphing


def algogradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


def drawLine(w, b, color):
    w = w / 100
    line1point = [55, (((w * 55) + b) + 55)]
    line2point = [790 + 55, ((w * 790) + b) + 55]
    pygame.draw.line(screen, color, line1point, line2point, width=5)


def drawPoints():
    for point in points:
        pygame.draw.circle(screen, BLACK, (point[0] + 55, point[1] + 55), 6)
        pygame.draw.circle(screen, WHITE, (point[0] + 55, point[1] + 55), 5)


def run():
    for point in points:
        listOfDistance = []
        for cluster in clusters:
            listOfDistance.append(distanceOfTwoPoints(point, cluster))
        labels.append(listOfDistance.index(min(listOfDistance)))

    for i in range(k):
        sumX = 0
        sumY = 0
        count = 0
        for j in range(len(points)):
            if labels[j] == i:
                sumX += points[j][0]
                sumY += points[j][1]
                count += 1
        if count != 0:
            clusters[i] = [int(sumX / count), int(sumY / count)]


def drawError(str):
    screen.blit(render_Errortext(str, 20), (200, 260))


def calculateErrorNumber():
    global error_number
    error_number = 0
    if clusters != [] and labels != []:
        for i in range(len(points)):
            error_number += distanceOfTwoPoints(points[i], clusters[labels[i]])


# Config:
running = True
w_init = 0
b_init = 0
tmp_alpha = 5.0e-2
num_iters = 0
w_final = w_init
b_final = b_init
J_history = []
p_history = []
error1 = False

while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    clock.tick(60)
    screen.fill(BACKGROUND)
    drawPanel()
    drawInterface()

    # End draw interface
    drawPoints()
    drawLine(w_init, b_init, RED)
    drawLine(w_final, b_final, BLUE)
    # Check event if it is trying to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if 860 < mouse_x < 1010 and 190 < mouse_y < 240:
                try:
                    w_final, b_final, J_history, p_history, num_iters = gradient_descent(x_train, y_train, w_final,
                                                                                         b_final, tmp_alpha,
                                                                                         compute_cost, compute_gradient,
                                                                                         num_iters, J_history,
                                                                                         p_history)
                except:
                    error1 = True
                print("RUN pressed")
            if 860 < mouse_x < 1060 and 260 < mouse_y < 310:
                try:
                    w_final, b_final, J_history, p_history = algogradient_descent(x_train, y_train, w_init, b_init,
                                                                                  tmp_alpha,
                                                                                  10000, compute_cost, compute_gradient)
                except:
                    error1 = True
                print("ALGORITHM pressed")
            if 55 < mouse_x < 845 and 55 < mouse_y < 545:
                labels = []
                point = [mouse_x - 55, mouse_y - 55]
                points.append(point)
                try:
                    x_train = np.append(x_train, (point[0] / 100))
                    y_train = np.append(y_train, (point[1]))
                except:
                    x_train = np.array([point[0] / 100])
                    y_train = np.array([point[1]])
                error1 = False
                print("ADD POINTS (" + str(mouse_x - 55) + ", " + str(mouse_y - 55) + ")")
    # Update what happen to the windows (pygame screen)
    if error1 == True:
        drawError("**Please add points first.")
    pygame.display.flip()

pygame.quit()
