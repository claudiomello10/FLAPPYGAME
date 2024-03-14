import sys, pygame, random
from multiprocessing import Pool

import tensorflow as tf
import numpy as np
from keras.layers import Dense
from tensorflow_probability.python.optimizer import differential_evolution_minimize


#   RUN VARIABLE                # CODE EXECUTION

#   "TRAIN_FROM_SCRATCH"        # Train the model from scratch
#   "TRAIN_WITH_LAST_RESULTS"   # Train the model with the last results
#   "TRAIN_WITH_BEST_BIRDS"     # Train the model with the best birds
#   "RUN_WITH_LAST_RESULTS"     # Run the model with the last results
#   "RUN_BEST_RESULT"           # Run the model with the best result
#   "RUN_WITH_BEST_BIRDS"       # Run the model with the best birds

if tf.test.gpu_device_name():
    print("GPU encontrado")
else:
    print("GPU não encontrado")

RUN = "TRAIN_WITH_BEST_BIRDS"

BROAD_ITERATIONS = 5
ITERATIONS = 50
POPULATION_SIZE = 10
DIFERENTIAL_WEIGHT = 2
CROSSOVER_PROB = 1


NEURAL_NETWORK_SHAPE = (7, 4, 2, 1)


X_NORMALIZE = 700
Y_NORMALIZE = 612

iteration_counter = 0


class Best_Birds:
    def __init__(self, weights, score):
        self.weights = weights
        self.score = score


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


previous_best = np.load("best_birds.npy", allow_pickle=True).tolist()


BEST = []
if previous_best.__len__() > 0:
    BEST = previous_best


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
    global BEST, iteration_counter
    weight_list = np.array(weight_list) if weight_list is not None else None

    if weight_list is not None:
        birds = []
        for i in range(weight_list.shape[0]):
            birds.append(Birds(weight_list[i]))
    else:
        birds = [Birds() for _ in range(players)]
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
    scores = tf.convert_to_tensor(
        np.array([bird.score for bird in birds]), dtype=tf.float32
    )

    for bird in birds:
        BEST.append(Best_Birds(bird.weights, bird.score))

    BEST.sort(key=lambda x: x.score)
    BEST = BEST[:100]
    iteration_counter += 1

    print(
        f"Best: {np.min(scores)*-1} Mean: {np.mean(scores)*-1} Iteration: {iteration_counter}/{ITERATIONS}"
    )
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


initial_weights = []
for i in range(POPULATION_SIZE):
    layer1K = np.random.uniform(
        low=-1.0, high=1.0, size=(NEURAL_NETWORK_SHAPE[0], NEURAL_NETWORK_SHAPE[1])
    ).flatten()
    layer1B = np.random.uniform(
        low=-1.0, high=1.0, size=(NEURAL_NETWORK_SHAPE[1])
    ).flatten()
    layer2K = np.random.uniform(
        low=-1.0, high=1.0, size=(NEURAL_NETWORK_SHAPE[1], NEURAL_NETWORK_SHAPE[2])
    ).flatten()
    layer2B = np.random.uniform(
        low=-1.0, high=1.0, size=(NEURAL_NETWORK_SHAPE[2])
    ).flatten()
    layer3K = np.random.uniform(
        low=-1.0, high=1.0, size=(NEURAL_NETWORK_SHAPE[2], NEURAL_NETWORK_SHAPE[3])
    ).flatten()
    layer3B = np.random.uniform(
        low=-1.0, high=1.0, size=(NEURAL_NETWORK_SHAPE[3])
    ).flatten()
    # Concatenate all flattened arrays into a single array
    weights = np.concatenate([layer1K, layer1B, layer2K, layer2B, layer3K, layer3B])
    initial_weights.append(weights)

# Convert the list of arrays into a 2D array
initial_weights = np.array(initial_weights)

# Convert the 2D array into a Tensor
initial_weights_tensor = tf.convert_to_tensor(initial_weights)


if RUN == "TRAIN_FROM_SCRATCH":

    print(
        f"\nTraining from scratch\n\nIterations: {ITERATIONS}\n\nPopulation size: {POPULATION_SIZE}\n\nDifferential weight: {DIFERENTIAL_WEIGHT}\n\nCrossover probability: {CROSSOVER_PROB}\n"
    )

    final_result = differential_evolution_minimize(
        run,
        initial_population=initial_weights_tensor,
        population_size=POPULATION_SIZE,
        max_iterations=ITERATIONS,
        differential_weight=DIFERENTIAL_WEIGHT,
    )

    np.savez("final_result.npz", *final_result)
    np.save("best_birds.npy", BEST)

else:
    final_result = np.load("final_result.npz", allow_pickle=True)
    final_result = [final_result[key] for key in final_result.files]
    converged = final_result[0]
    objective_evaluations = final_result[1]
    best_weights = final_result[2]
    best_scores = final_result[3]
    final_weights = final_result[4]
    final_scores = final_result[5]
    initial_weights = final_result[6]
    initial_scores = final_result[7]
    number_of_iterations = final_result[8]

    if RUN == "TRAIN_WITH_LAST_RESULTS":

        print(
            f"\nTraining with last results\n\nIterations: {ITERATIONS}\n\nPopulation size: {final_weights.shape[0]}\n\nDifferential weight: {DIFERENTIAL_WEIGHT}\n\nCrossover probability: {CROSSOVER_PROB}\n"
        )

        final_result = differential_evolution_minimize(
            run,
            initial_population=final_weights,
            population_size=final_weights.shape[0],
            max_iterations=ITERATIONS,
            differential_weight=DIFERENTIAL_WEIGHT,
            crossover_prob=CROSSOVER_PROB,
        )
        np.savez("final_result.npz", *final_result)
        np.save("best_birds.npy", BEST)

    elif RUN == "TRAIN_WITH_BEST_BIRDS":
        best_birds_weights = np.array([bird.weights for bird in BEST])
        best_birds_scores = np.array([bird.score for bird in BEST])
        print(best_birds_weights.shape[0])

        print(
            f"\nTraining with best birds\n\nIterations: {ITERATIONS}\n\nPopulation size: {best_birds_weights.shape[0]}\n\nDifferential weight: {DIFERENTIAL_WEIGHT}\n\nCrossover probability: {CROSSOVER_PROB}\n\nPrevious best score: {np.min(best_birds_scores)*-1}\n\n"
        )
        for i in range(BROAD_ITERATIONS):
            iteration_counter = 0
            print(f"Running broad iteration {i+1}/{BROAD_ITERATIONS}\n")
            best_birds_weights = np.array([bird.weights for bird in BEST])
            final_result = differential_evolution_minimize(
                run,
                initial_population=best_birds_weights,
                population_size=best_birds_weights.shape[0],
                max_iterations=ITERATIONS,
                differential_weight=DIFERENTIAL_WEIGHT,
                crossover_prob=CROSSOVER_PROB,
            )
            np.savez("final_result.npz", *final_result)
            np.save("best_birds.npy", BEST)

    elif RUN == "RUN_WITH_BEST_BIRDS":
        best_birds = np.load("best_birds.npy", allow_pickle=True)
        best_birds_scores = np.array([bird.score for bird in best_birds])
        best_birds_weights = np.array([bird.weights for bird in best_birds])
        print(
            f"Running with best birds\n\nPopulation size: {best_birds_weights.shape[0]}\n\nBest score: {np.min(best_birds_scores)*-1}\n\nMean score: {np.mean(best_birds_scores)*-1}\n\n"
        )
        run(best_birds_weights)

    elif RUN == "RUN_WITH_LAST_RESULTS":
        print(
            f"Running with last results\n\nPopulation size: {final_weights.shape[0]}\n\nBest score: {np.min(final_scores)*-1}\n\nMean score: {np.mean(final_scores)*-1}\n\n"
        )
        run(final_weights)
    elif RUN == "RUN_BEST_RESULT":
        print(
            f"Running with best result\n\nBest score: {np.min(best_scores)*-1}\n\nMean score: {np.mean(best_scores)*-1}\n\n"
        )
        np_best_weights = best_weights.numpy().reshape(
            1,
            NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
            + NEURAL_NETWORK_SHAPE[1]
            + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
            + NEURAL_NETWORK_SHAPE[2]
            + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
            + NEURAL_NETWORK_SHAPE[3],
        )
        run(np_best_weights)
    else:
        print("Invalid run mode")
        exit(1)
