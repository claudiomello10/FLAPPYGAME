import sys, pygame, random

import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model
from tensorflow_probability.python.optimizer import differential_evolution_minimize


    #   RUN VARIABLE                # CODE EXECUTION

    #   "TRAIN_FROM_SCRATCH"        # Train the model from scratch
    #   "TRAIN_WITH_LAST_RESULTS"   # Train the model with the last results
    #   "RUN_WITH_LAST_RESULTS"     # Run the model with the last results
    #   "RUN_BEST_RESULT"           # Run the model with the best result



RUN = "TRAIN_WITH_LAST_RESULTS"



ITERATIONS = 500
POPULATION_SIZE = 100
NEURAL_NETWORK_SHAPE = (26, 5, 5, 1)




class Birds:

    def calculate_output(self, inputs):
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
                5,
                activation="sigmoid",
            ),
            Dense(5, activation="sigmoid"),
            Dense(1, activation="sigmoid"),
        ]
        layers[0].build((1, 26))
        layers[1].build((1, 5))
        layers[2].build((1, 5))


        if weights is not None:
            current = 0
            weight0 = weights[:26 * 5].reshape(26, 5)
            current += 26 * 5
            bias0 = weights[current:current + 5].reshape(5)
            current += 5
            weight1 = weights[current:current + 5 * 5].reshape(5, 5)
            current += 5 * 5
            bias1 = weights[current:current + 5].reshape(5)
            current += 5
            weight2 = weights[current:current + 5].reshape(
                5, 1
            )
            current += 5
            bias2 = weights[current:current + 1].reshape(
                1
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
    return bottom_pipe, top_pipe


def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 4
    return pipes


def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface, pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe)


def check_collision(pipes, bird_rect):
    for pipe in pipes:
        if (
            bird_rect.colliderect(pipe)

        ):
            return (False, False)
        if(           bird_rect.top <= -100
            or bird_rect.bottom >= 900):
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


pygame.mixer.pre_init(frequency=44100, size=32, channels=1, buffer=512)
pygame.init()

screen = pygame.display.set_mode((576, 1024))
clock = pygame.time.Clock()
game_font = pygame.font.Font("FlappyBirdy.ttf", 72)

# PLEU
players = 1


# VARIAVEIS
gravity = 0.25

# SOUND
flap_sound = pygame.mixer.Sound("audio/wing.wav")


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


def run(weight_list=None):
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
        pipe_parameters = np.asarray(pipe_list).flatten()
        np.pad(
            pipe_parameters,
            (0, 24 - pipe_parameters.size),
            "constant",
            constant_values=(0, 0),
        )

        for Bird in birds:
            if Bird.alive:
                
                specific_parameters = np.pad(
                    pipe_parameters/100,
                    (1, 24 - pipe_parameters.size),
                    "constant",
                    constant_values=(Bird.movement/12, 0),
                )
                specific_parameters = np.append(specific_parameters, Bird.rect.centery/1024)
                specific_parameters = specific_parameters.reshape(1, 26)
                result = Bird.calculate_output(specific_parameters)
                
                if result == True:
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
        if (pipe_list.__len__() > 0 and pipe_list[-1].centerx < 200) or (
            pipe_list.__len__() == 0
        ):
            if pipe_list.__len__() >= 6:
                pipe_list.pop(0)
                pipe_list.pop(0)
            pipe_list.extend(create_pipe())
            waiting = True
            print("waiting")
            
            

        if(waiting):
            wait += 1
            if(wait > 20):
                waiting = False
                wait = 0
                for Bird in birds:
                    if Bird.alive:
                        Bird.score -= 20
                if game_active == True:
                    score += 1


        if not any(bird.alive for bird in birds):
            game_active = False

        if game_active:
            
            for Bird in birds:
                if Bird.alive:
                    
                    Bird.score -= 1
                    (alive, penalidade) = check_collision(pipe_list, Bird.rect)
                    if(penalidade):
                        Bird.score += 10
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
        clock.tick(120)
    scores = tf.convert_to_tensor(np.array([bird.score for bird in birds]), dtype=tf.float32)
    print(scores)
    return scores



initial_weights = []
for i in range(POPULATION_SIZE):
    layer1K = np.random.uniform(low=-1.0, high=1.0, size=(26, 5)).flatten()
    layer1B = np.random.uniform(low=-1.0, high=1.0, size=(5,1)).flatten()
    layer2K = np.random.uniform(low=-1.0, high=1.0, size=(5,5)).flatten()
    layer2B = np.random.uniform(low=-1.0, high=1.0, size=(5,1)).flatten()
    layer3K = np.random.uniform(low=-1.0, high=1.0, size=(5,1)).flatten()
    layer3B = np.random.uniform(low=-1.0, high=1.0, size=(1)).flatten()
    # Concatenate all flattened arrays into a single array
    weights = np.concatenate([layer1K, layer1B, layer2K, layer2B, layer3K, layer3B])
    initial_weights.append(weights)

# Convert the list of arrays into a 2D array
initial_weights = np.array(initial_weights)

# Convert the 2D array into a Tensor
initial_weights_tensor = tf.convert_to_tensor(initial_weights)








if (RUN == "TRAIN_FROM_SCRATCH"):

    final_result = differential_evolution_minimize(
        run,
        initial_population=initial_weights_tensor,
        population_size=POPULATION_SIZE,
        max_iterations=ITERATIONS,
        differential_weight=1.5,
    )

    np.save("final_result.npy", final_result)

else:
    final_result = np.load("final_result.npy", allow_pickle=True)
    converged = final_result[0]
    objective_evaluations = final_result[1]
    best_weights = final_result[2]
    best_scores = final_result[3]
    final_weights = final_result[4]
    final_scores = final_result[5]
    initial_weights = final_result[6]
    initial_scores = final_result[7]
    number_of_iterations = final_result[8]


    if(RUN == "TRAIN_WITH_LAST_RESULTS"):


        final_result = differential_evolution_minimize(
            run,
            initial_population=final_weights,
            population_size=POPULATION_SIZE,
            max_iterations=ITERATIONS,
            differential_weight=1.5,
        )
        np.save("final_result.npy", final_result)


    elif(RUN == "RUN_WITH_LAST_RESULTS"):
        run(final_weights)
    elif(RUN == "RUN_BEST_RESULT"):
        np_best_weights = best_weights.numpy().reshape(1,NEURAL_NETWORK_SHAPE[0]*NEURAL_NETWORK_SHAPE[1]+NEURAL_NETWORK_SHAPE[1]+NEURAL_NETWORK_SHAPE[1]*NEURAL_NETWORK_SHAPE[2]+NEURAL_NETWORK_SHAPE[2]+NEURAL_NETWORK_SHAPE[2]*NEURAL_NETWORK_SHAPE[3]+NEURAL_NETWORK_SHAPE[3])
        run(np_best_weights)
    else:
        print("Invalid run mode")
        exit(1)

