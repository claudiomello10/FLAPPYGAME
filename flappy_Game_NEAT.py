import sys, pygame, random
import torch
from torch import nn
import numpy as np
import pygad
import inquirer

# Check if the GPU is available
if torch.cuda.is_available():
    print("Torch está usando a GPU.")
else:
    print("Torch está usando a CPU.")


questions = [
    inquirer.List(
        "run_mode",
        message="Select run mode:",
        choices=[
            ("Train the model from scratch", "TRAIN_FROM_SCRATCH"),
            ("Train the model with the last results", "TRAIN_WITH_LAST_RESULTS"),
            ("Run the model with the best result", "RUN_BEST_RESULT"),
        ],
    ),
]

answers = inquirer.prompt(questions)

# Define the hyperparameters
GENERATIONS = 10
POPULATION_SIZE = 20
NEURAL_NETWORK_SHAPE = (7, 4, 2, 1)


# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, weights):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(NEURAL_NETWORK_SHAPE[0], NEURAL_NETWORK_SHAPE[1])
        self.fc2 = nn.Linear(NEURAL_NETWORK_SHAPE[1], NEURAL_NETWORK_SHAPE[2])
        self.fc3 = nn.Linear(NEURAL_NETWORK_SHAPE[2], NEURAL_NETWORK_SHAPE[3])
        if weights is not None:
            self.fc1.weight.data = torch.from_numpy(
                weights[: NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]]
            ).reshape(NEURAL_NETWORK_SHAPE[1], NEURAL_NETWORK_SHAPE[0])
            self.fc1.bias.data = torch.from_numpy(
                weights[
                    NEURAL_NETWORK_SHAPE[0]
                    * NEURAL_NETWORK_SHAPE[1] : NEURAL_NETWORK_SHAPE[0]
                    * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                ]
            )

            self.fc2.weight.data = torch.from_numpy(
                weights[
                    NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1] : NEURAL_NETWORK_SHAPE[0]
                    * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
                ]
            ).reshape(NEURAL_NETWORK_SHAPE[2], NEURAL_NETWORK_SHAPE[1])
            self.fc2.bias.data = torch.from_numpy(
                weights[
                    NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    * NEURAL_NETWORK_SHAPE[2] : NEURAL_NETWORK_SHAPE[0]
                    * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
                    + NEURAL_NETWORK_SHAPE[2]
                ]
            )

            self.fc3.weight.data = torch.from_numpy(
                weights[
                    NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
                    + NEURAL_NETWORK_SHAPE[2] : NEURAL_NETWORK_SHAPE[0]
                    * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
                    + NEURAL_NETWORK_SHAPE[2]
                    + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
                ]
            ).reshape(NEURAL_NETWORK_SHAPE[3], NEURAL_NETWORK_SHAPE[2])
            self.fc3.bias.data = torch.from_numpy(
                weights[
                    NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1]
                    + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
                    + NEURAL_NETWORK_SHAPE[2]
                    + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3] :
                ]
            )

    def forward(self, x):
        x = torch.flatten(x, 1).double()  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = torch.sigmoid(self.fc3(x))  # Output layer (no activation function)
        return x


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
        inputs = np.array(inputs).reshape(1, NEURAL_NETWORK_SHAPE[0])
        inputs = torch.from_numpy(inputs).double()
        output = self.model.forward(inputs)

        if output > 0.9:
            return True
        else:
            return False

    def create_model(self, weights=None):
        model = NeuralNetwork(weights)
        initialized_weights = np.array([])

        return model

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
        self.model = self.create_model(weights)


def draw_floor(floor_x_pos, screen, floor_surface):

    screen.blit(floor_surface, (floor_x_pos, 900))
    screen.blit(floor_surface, (floor_x_pos + 576, 900))


def create_pipe(pipe_surface):
    random_pipe_pos = random.randrange(400, 800, 100)
    bottom_pipe = pipe_surface.get_rect(midtop=(700, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 300))
    return [[bottom_pipe, top_pipe]]


def move_pipes(pipes):
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            pipe.centerx -= 4
    return pipes


def draw_pipes(pipes, screen, pipe_surface):
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


def score_display(score, game_font, screen):
    if score < 0:
        score_surface = game_font.render(str(int(0)), True, (255, 255, 255))
    else:
        score_surface = game_font.render(str(int(score)), True, (255, 255, 255))
    score_rect = score_surface.get_rect(center=(288, 100))
    screen.blit(score_surface, score_rect)


def run(weight_list=None, slow=False, fps=60):
    screen, game_font, floor_surface, bg_surface, BIRDFLAP, pipe_surface = (
        initialize_game()
    )
    if weight_list is not None:
        birds = [Birds(weight_list)]
    else:
        birds = [Birds()]
    gravity = 0.25
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
            (0, 4 - pipe_parameters.size),
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
            pipe_list.extend(create_pipe(pipe_surface))
            waiting = True

        if waiting:
            wait += 1
            if wait > 20:
                waiting = False
                wait = 0
                for Bird in birds:
                    if Bird.alive:
                        Bird.score += 60
                if game_active == True:
                    score += 1

        if not any(bird.alive for bird in birds):
            game_active = False

        if game_active:

            for Bird in birds:
                if Bird.alive:

                    Bird.score += 1
                    (alive, penalidade) = check_collision(pipe_list, Bird.rect)
                    if penalidade:
                        Bird.score -= 1000
                    Bird.alive = alive
                    Bird.movement += gravity
                    rotated_bird = rotate_bird(Bird.surface, Bird.movement)
                    Bird.rect.centery += Bird.movement
                    screen.blit(rotated_bird, Bird.rect)

            # PIPES
            pipe_list = move_pipes(pipe_list)
            draw_pipes(pipe_list, screen, pipe_surface)
            score_display(score, game_font, screen)
            if slow:
                pygame.time.Clock().tick(fps)

        # FLOOR
        if game_active:
            floor_x_pos -= 4
        draw_floor(floor_x_pos, screen, floor_surface)
        if floor_x_pos <= -576:
            floor_x_pos = 0
        pygame.display.update()
    scores = [bird.score for bird in birds]

    return scores


def initialize_game():
    # pygame.mixer.pre_init(frequency=44100, size=32, channels=1, buffer=512)
    pygame.init()
    screen = pygame.display.set_mode((576, 1024))
    game_font = pygame.font.Font("FlappyBirdy.ttf", 72)

    # SURFACES
    floor_surface = pygame.image.load("sprites/base.png").convert()
    floor_surface = pygame.transform.scale2x(floor_surface)

    bg_surface = pygame.image.load("sprites/background-day.png").convert()
    bg_surface = pygame.transform.scale2x(bg_surface)

    BIRDFLAP = pygame.USEREVENT + 1
    pygame.time.set_timer(BIRDFLAP, 200)

    pipe_surface = pygame.image.load("sprites/pipe-green.png").convert()
    pipe_surface = pygame.transform.scale2x(pipe_surface)

    return screen, game_font, floor_surface, bg_surface, BIRDFLAP, pipe_surface


# Define callback generation function
def callback_generation(ga_instance):
    print(
        f"Generation: {ga_instance.generations_completed} ------------------ Score: {ga_instance.best_solution()[1]}"
    )


# Define the fitness function


def fitness_func(ga_instance, solution, solution_idx):
    # Convert the solution to numpy array
    weights = np.array(solution)
    # Run the game and get the scores
    scores = run(weights)

    # Return the negative scores as the fitness values
    return scores


def train_from_scratch():
    # Create an instance of the pygad.GA class

    generations = input("Enter the number of generations: ")
    try:
        generations = int(generations)
    except Exception as e:
        print(f"\n{generations} is not a valid integer for number of generations")
        print(f"\nRunning with default value {GENERATIONS}\n")
        generations = GENERATIONS

    population_size = input("Enter the population size: ")
    try:
        population_size = int(population_size)
    except Exception as e:
        print(f"\n{population_size} is not a valid integer for population size")
        print(f"\nRunning with default value {POPULATION_SIZE}\n")
        population_size = POPULATION_SIZE

    ga_instance = pygad.GA(
        sol_per_pop=population_size,
        num_genes=NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
        + NEURAL_NETWORK_SHAPE[1]
        + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
        + NEURAL_NETWORK_SHAPE[2]
        + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
        + NEURAL_NETWORK_SHAPE[3],
        num_generations=generations,
        num_parents_mating=3,
        keep_parents=1,
        on_generation=callback_generation,
        fitness_func=fitness_func,
    )

    # Start the genetic algorithm optimization
    ga_instance.run()

    # Get the best solution and its fitness value
    best_solution = ga_instance.best_solution()

    (best_weights, best_score, index) = best_solution

    np.save("best_birds.npy", best_weights)
    try:
        ga_instance.plot_fitness()
    except Exception as e:
        print("Error plotting the fitness graph")


def train_with_last_results():
    best_bird = np.load("best_birds.npy", allow_pickle=True)
    print(best_bird)

    generations = input("Enter the number of generations: ")
    try:
        generations = int(generations)
    except Exception as e:
        print(f"\n{generations} is not a valid integer for number of generations")
        print(f"\nRunning with default value {GENERATIONS}\n")
        generations = GENERATIONS

    population_size = input("Enter the population size: ")
    try:
        population_size = int(population_size)
    except Exception as e:
        print(f"\n{population_size} is not a valid integer for population size")
        print(f"\nRunning with default value {POPULATION_SIZE}\n")
        population_size = POPULATION_SIZE

    initial_population = np.array([best_bird for _ in range(POPULATION_SIZE)])

    print(
        f"\nTraining with last results\n\nIterations: {GENERATIONS}\n\nPopulation size: {POPULATION_SIZE}\n\n"
    )

    ga_instance = pygad.GA(
        initial_population=initial_population,
        num_genes=NEURAL_NETWORK_SHAPE[0] * NEURAL_NETWORK_SHAPE[1]
        + NEURAL_NETWORK_SHAPE[1]
        + NEURAL_NETWORK_SHAPE[1] * NEURAL_NETWORK_SHAPE[2]
        + NEURAL_NETWORK_SHAPE[2]
        + NEURAL_NETWORK_SHAPE[2] * NEURAL_NETWORK_SHAPE[3]
        + NEURAL_NETWORK_SHAPE[3],
        num_generations=generations,
        num_parents_mating=3,
        keep_parents=1,
        on_generation=callback_generation,
        fitness_func=fitness_func,
    )

    ga_instance.run()
    best_solution = ga_instance.best_solution()
    (best_weights, best_score, index) = best_solution

    np.save("best_birds.npy", best_weights)
    try:
        ga_instance.plot_fitness()
    except Exception as e:
        print("Error plotting the fitness graph")


def run_best_result():
    fps = input("\nEnter the desired max FPS: ")

    try:

        fps = int(fps)
    except Exception as e:
        print(f"\n{fps} is not a valid integer for max FPS")
        print("\nRunning with default value 60\n")
        fps = 60

    best_birds = np.load("best_birds.npy")

    print(f"Running with best result\n\n")

    result = run(best_birds, slow=True, fps=fps)

    print(f"Score: {result}")


if __name__ == "__main__":
    run_mode = answers["run_mode"]

    if run_mode == "TRAIN_FROM_SCRATCH":
        train_from_scratch()
    elif run_mode == "TRAIN_WITH_LAST_RESULTS":
        train_with_last_results()
    elif run_mode == "RUN_BEST_RESULT":
        run_best_result()
    else:
        print("Invalid run mode")
        exit(1)
