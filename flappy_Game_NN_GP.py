import sys, pygame, random
from multiprocessing import Pool

import tensorflow as tf
import numpy as np
from keras.layers import Dense
from skopt import gp_minimize


if tf.test.gpu_device_name():
    print("GPU encontrado")
else:
    print("GPU não encontrado")


NEURAL_NETWORK_SHAPE = (7, 4, 2, 1)

RANGE = (-10000.0, 10000.0)


DIMENSIONS = [
    RANGE
    for _ in range(
        NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
        + NEURAL_NETWORK_SHAPE[1]
        + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
        + NEURAL_NETWORK_SHAPE[2]
        + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
        + NEURAL_NETWORK_SHAPE[3]
    )
]


iteration_counter = 0


class Birds:

    def calculate_output(self, inputs):
        inputs = np.append(
            inputs,
            [
                self.movement,
                self.rect.centery,
                self.rect.centerx,
            ],
        )
        inputs = np.reshape(inputs, (1, NEURAL_NETWORK_SHAPE[0]))
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        aux1 = self.layers[0](inputs)
        aux2 = self.layers[1](aux1)
        aux3 = self.layers[2](aux2)
        if aux3 > 0.5:
            return True
        else:
            return False

    def create_layers(self, weights):
        layers = [
            Dense(
                NEURAL_NETWORK_SHAPE[1],
                activation="relu",
            ),
            Dense(NEURAL_NETWORK_SHAPE[2], activation="relu"),
            Dense(NEURAL_NETWORK_SHAPE[3], activation="sigmoid"),
        ]
        layers[0].build((1, NEURAL_NETWORK_SHAPE[0]))
        layers[1].build((1, NEURAL_NETWORK_SHAPE[1]))
        layers[2].build((1, NEURAL_NETWORK_SHAPE[2]))

        initialized_weights = np.array([])
        for layer in layers:
            initialized_weights = np.append(
                initialized_weights, layer.get_weights()[0].flatten()
            )
            initialized_weights = np.append(
                initialized_weights, layer.get_weights()[1].flatten()
            )
        self.weights = initialized_weights

        if weights is not None:
            self.weights = weights
            current = 0
            weight0 = weights[
                : NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
            ].reshape(NEURAL_NETWORK_SHAPE[0], NEURAL_NETWORK_SHAPE[1])
            current += NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
            bias0 = weights[current : current + NEURAL_NETWORK_SHAPE[1]].reshape(
                NEURAL_NETWORK_SHAPE[1]
            )
            current += NEURAL_NETWORK_SHAPE[1]
            weight1 = weights[
                current : current + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
            ].reshape(NEURAL_NETWORK_SHAPE[1], NEURAL_NETWORK_SHAPE[2])
            current += NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
            bias1 = weights[current : current + NEURAL_NETWORK_SHAPE[2]].reshape(
                NEURAL_NETWORK_SHAPE[2]
            )
            current += NEURAL_NETWORK_SHAPE[2]
            weight2 = weights[
                current : current + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
            ].reshape(NEURAL_NETWORK_SHAPE[2], NEURAL_NETWORK_SHAPE[3])
            current += NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
            bias2 = weights[current : current + NEURAL_NETWORK_SHAPE[3]].reshape(
                NEURAL_NETWORK_SHAPE[3]
            )
            layers[0].set_weights([weight0, bias0])
            layers[1].set_weights([weight1, bias1])
            layers[2].set_weights([weight2, bias2])

        return layers

    def __init__(self, weights=None):
        self.downflap = pygame.transform.scale2x(
            pygame.image.load("sprites/bluebird-downflap.png").convert_alpha()
        )
        self.midflap = pygame.transform.scale2x(
            pygame.image.load("sprites/bluebird-midflap.png").convert_alpha()
        )
        self.upflap = pygame.transform.scale2x(
            pygame.image.load("sprites/bluebird-upflap.png").convert_alpha()
        )
        self.frames = [self.downflap, self.midflap, self.upflap]
        self.index = 2
        self.surface = self.frames[self.index]
        self.rect = self.surface.get_rect(center=(100, 512))
        self.movement = 0
        self.score = 0
        self.alive = True
        self.layers = self.create_layers(weights)


def draw_floor(floor_x_pos):

    screen.blit(floor_surface, (floor_x_pos, 900))
    screen.blit(floor_surface, (floor_x_pos + 576, 900))


def create_pipe():
    random_pipe_pos = random.randrange(400, 800, 100)
    bottom_pipe = pipe_surface.get_rect(midtop=(700, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 300))
    return [[bottom_pipe, top_pipe]]


def move_pipes(pipes):
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            pipe.centerx -= 4
    return pipes


def draw_pipes(pipes):
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            if pipe.bottom >= 1024:
                screen.blit(pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(pipe_surface, False, True)
                screen.blit(flip_pipe, pipe)


def check_collision(pipes, bird_rect):
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            if bird_rect.colliderect(pipe):
                return (False, False)
            if bird_rect.top <= -100 or bird_rect.bottom >= 900:
                return (False, True)
    return (True, False)


def rotate_bird(bird, bird_movement):
    new_bird = pygame.transform.rotozoom(bird, -bird_movement * 3, 1)
    return new_bird


def bird_animation(bird_frames, bird_index, bird_rect):
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center=(100, bird_rect.centery))
    return new_bird, new_bird_rect


def score_display(score):
    if score < 0:
        score_surface = game_font.render(str(int(0)), True, (255, 255, 255))
    else:
        score_surface = game_font.render(str(int(score)), True, (255, 255, 255))
    score_rect = score_surface.get_rect(center=(288, 100))
    screen.blit(score_surface, score_rect)


def run(weight_list=None):
    weight_list = np.array(weight_list) if weight_list is not None else None

    if weight_list is not None:
        birds = [Birds(weight_list)]
    else:
        birds = [Birds()]
    floor_x_pos = 0
    pipe_list = []
    game_active = True
    score = 0
    wait = 0
    waiting = False
    while game_active == True:
        pipe_parameters = []
        for pipe in pipe_list:
            if pipe[0].centerx > 0:
                pipe_parameters.append(
                    [
                        (pipe[0].centerx + pipe[1].centerx) / 2,
                        pipe[0].centery + pipe[1].centery,
                    ]
                )
        if pipe_parameters.__len__() > 2:
            pipe_parameters = pipe_parameters[0:2]

        pipe_parameters = np.asarray(pipe_parameters).flatten()
        pipe_parameters = np.pad(
            pipe_parameters,
            (4 - pipe_parameters.size, 0),
            "constant",
            constant_values=(0, 0),
        )

        for Bird in birds:
            if Bird.alive:
                if Bird.calculate_output(pipe_parameters):
                    Bird.movement = 0
                    Bird.movement -= 10

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == BIRDFLAP:
                for Bird in birds:
                    if Bird.alive:
                        if Bird.index < 2:
                            Bird.index += 1
                        else:
                            Bird.index = 0
                        Bird.surface, Bird.rect = bird_animation(
                            Bird.frames, Bird.index, Bird.rect
                        )

        screen.blit(bg_surface, (0, 0))
        if (pipe_list.__len__() > 0 and pipe_list[-1][0].centerx < 200) or (
            pipe_list.__len__() == 0
        ):
            if pipe_list.__len__() >= 3:
                pipe_list.pop(0)
            pipe_list.extend(create_pipe())
            waiting = True

        if waiting:
            wait += 1
            if wait > 20:
                waiting = False
                wait = 0
                for Bird in birds:
                    if Bird.alive:
                        Bird.score -= 60
                if game_active == True:
                    score += 1

        if not any(bird.alive for bird in birds):
            game_active = False

        if game_active:

            for Bird in birds:
                if Bird.alive:

                    Bird.score -= 1
                    (alive, penalidade) = check_collision(pipe_list, Bird.rect)
                    if penalidade:
                        Bird.score += 1000
                    Bird.alive = alive
                    Bird.movement += gravity
                    rotated_bird = rotate_bird(Bird.surface, Bird.movement)
                    Bird.rect.centery += Bird.movement
                    screen.blit(rotated_bird, Bird.rect)

            # PIPES
            pipe_list = move_pipes(pipe_list)
            draw_pipes(pipe_list)
            score_display(score)

        # FLOOR
        if game_active:
            floor_x_pos -= 4
        draw_floor(floor_x_pos)
        if floor_x_pos <= -576:
            floor_x_pos = 0
        pygame.display.update()
        scores = birds[0].score

    print(f"Score: {scores}")
    return scores


# pygame.mixer.pre_init(frequency=44100, size=32, channels=1, buffer=512)
pygame.init()

screen = pygame.display.set_mode((576, 1024))
game_font = pygame.font.Font("FlappyBirdy.ttf", 72)

# PLEU
players = 1


# VARIAVEIS
gravity = 0.25

# SOUND
# flap_sound = pygame.mixer.Sound("audio/wing.wav")


# SURFACES
floor_surface = pygame.image.load("sprites/base.png").convert()
floor_surface = pygame.transform.scale2x(floor_surface)


bg_surface = pygame.image.load("sprites/background-day.png").convert()
bg_surface = pygame.transform.scale2x(bg_surface)


BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP, 200)

# bird_surface = pygame.image.load("sprites/bluebird-midflap.png").convert_alpha()
# bird_surface = pygame.transform.scale2x(bird_surface)
# bird_rect = bird_surface.get_rect(center = (100,512))


pipe_surface = pygame.image.load("sprites/pipe-green.png").convert()
pipe_surface = pygame.transform.scale2x(pipe_surface)


result = gp_minimize(run, DIMENSIONS, n_calls=100, n_jobs=10)
print(result)
