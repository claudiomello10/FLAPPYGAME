"""
Flappy Bird Game with Neural Network and Genetic Algorithm

This project implements a Flappy Bird game controlled by a neural network trained using a genetic algorithm.
The neural network learns to play the game by evolving over multiple generations.

Modules:
- sys, pygame, random: For game implementation and control.
- torch, torch.nn: For neural network implementation.
- numpy: For numerical operations.
- pygad: For genetic algorithm implementation.
- inquirer: For command-line interface.

Classes:
- NeuralNetwork: Defines the neural network architecture.
- Best_Birds: Stores the best birds' weights and scores.
- Birds: Represents the bird in the game and its neural network model.

Functions:
- draw_floor: Draws the floor of the game.
- create_pipe: Creates a new pipe.
- move_pipes: Moves the pipes.
- draw_pipes: Draws the pipes.
- check_collision: Checks for collisions between the bird and pipes.
- rotate_bird: Rotates the bird based on its movement.
- bird_animation: Animates the bird's flapping.
- score_display: Displays the current score.
- run: Runs the game with the given neural network weights.
- initialize_game: Initializes the game environment.
- callback_generation: Callback function for each generation in the genetic algorithm.
- fitness_func: Fitness function for the genetic algorithm.
- train_from_scratch: Trains the neural network from scratch using the genetic algorithm.
- train_with_last_results: Continues training the neural network from the last saved results.
- run_best_result: Runs the game with the best neural network weights.

Usage:
Run the script and follow the command-line prompts to train or run the model.

Author: Cláudio Klautau Mello
"""

import sys, pygame, random
import torch
from torch import nn
import numpy as np
import pygad
import inquirer


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
        # Set the weights if provided
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
        """
        Calculate the output of the neural network based on the given inputs.
        This method appends additional features (movement, rect.centery, rect.centerx)
        to the input array, reshapes it to match the neural network's input shape,
        converts it to a PyTorch tensor, and then feeds it through the model to get
        the output.
        Args:
            inputs (list or np.ndarray): The initial input features for the neural network.
        Returns:
            bool: True if the output of the neural network is greater than 0.9, otherwise False.
        """
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
        """
        Creates and returns a NeuralNetwork model.
        Args:
            weights (optional): Initial weights for the neural network. Default is None.
        Returns:
            NeuralNetwork: An instance of the NeuralNetwork class.
        """
        model = NeuralNetwork(weights)
        initialized_weights = np.array([])

        return model

    def __init__(self, weights=None):
        """
        Initializes the bird object with optional neural network weights.

        Parameters:
        weights (list, optional): A list of weights for the neural network model. Defaults to None.

        Attributes:
        downflap (pygame.Surface): The surface for the bird's downflap sprite.
        midflap (pygame.Surface): The surface for the bird's midflap sprite.
        upflap (pygame.Surface): The surface for the bird's upflap sprite.
        frames (list): A list containing the bird's animation frames.
        index (int): The current frame index for the bird's animation.
        surface (pygame.Surface): The current surface of the bird based on the animation frame.
        rect (pygame.Rect): The rectangle representing the bird's position and size.
        movement (int): The current movement speed of the bird.
        score (int): The current score of the bird.
        alive (bool): A flag indicating whether the bird is alive.
        model (object): The neural network model created using the provided weights.
        """
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
    """
    Draws the floor surface on the screen at the specified x position.
    Args:
        floor_x_pos (int): The x position of the floor surface.
        screen (pygame.Surface): The screen surface where the floor will be drawn.
        floor_surface (pygame.Surface): The surface image of the floor to be drawn.
    Returns:
        None
    """

    screen.blit(floor_surface, (floor_x_pos, 900))
    screen.blit(floor_surface, (floor_x_pos + 576, 900))


def create_pipe(pipe_surface):
    """
    Create a pair of pipes (top and bottom) for the Flappy Bird game.

    Args:
        pipe_surface (pygame.Surface): The surface image of the pipe.

    Returns:
        list: A list containing two pygame.Rect objects representing the bottom and top pipes.
    """
    random_pipe_pos = random.randrange(400, 800, 100)
    bottom_pipe = pipe_surface.get_rect(midtop=(700, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 300))
    return [[bottom_pipe, top_pipe]]


def move_pipes(pipes):
    """
    Moves each pipe in the list of pipes to the left by 4 units.

    Args:
        pipes (list of list of pygame.Rect): A list where each element is a list containing two pygame.Rect objects representing the top and bottom pipes.

    Returns:
        list of list of pygame.Rect: The updated list of pipes with their positions moved to the left.
    """
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            pipe.centerx -= 4
    return pipes


def draw_pipes(pipes, screen, pipe_surface):
    """
    Draws the pipes on the screen.

    Args:
        pipes (list): A list of tuples, where each tuple contains two pipe Rect objects.
        screen (pygame.Surface): The surface on which to draw the pipes.
        pipe_surface (pygame.Surface): The surface representing the pipe image.

    Returns:
        None
    """
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            if pipe.bottom >= 1024:
                screen.blit(pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(pipe_surface, False, True)
                screen.blit(flip_pipe, pipe)


def check_collision(pipes, bird_rect):
    """
    Check for collisions between the bird and the pipes or screen boundaries.

    Args:
        pipes (list): A list of tuples, where each tuple contains two pipe rectangles (top and bottom).
        bird_rect (pygame.Rect): The rectangle representing the bird's current position.

    Returns:
        tuple: A tuple containing two boolean values:
            - The first boolean indicates whether the bird is still in play (True if no collision with pipes).
            - The second boolean indicates whether the bird has collided with the screen boundaries (True if collision with top or bottom boundary).
    """
    for pipe_couple in pipes:
        for pipe in pipe_couple:
            if bird_rect.colliderect(pipe):
                return (False, False)
            if bird_rect.top <= -100 or bird_rect.bottom >= 900:
                return (False, True)
    return (True, False)


def rotate_bird(bird, bird_movement):
    """
    Rotates the bird image based on its movement.

    Args:
        bird (pygame.Surface): The surface representing the bird image.
        bird_movement (float): The current movement of the bird, which determines the rotation angle.

    Returns:
        pygame.Surface: The new surface with the bird image rotated.
    """
    new_bird = pygame.transform.rotozoom(bird, -bird_movement * 3, 1)
    return new_bird


def bird_animation(bird_frames, bird_index, bird_rect):
    """
    Animates the bird by selecting the appropriate frame and updating its rectangle.

    Args:
        bird_frames (list): A list of bird frame images.
        bird_index (int): The index of the current bird frame.
        bird_rect (pygame.Rect): The rectangle of the current bird frame.

    Returns:
        tuple: A tuple containing the new bird frame (pygame.Surface) and its updated rectangle (pygame.Rect).
    """
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center=(100, bird_rect.centery))
    return new_bird, new_bird_rect


def score_display(score, game_font, screen):
    """
    Displays the current score on the game screen.

    Args:
        score (int): The current score to be displayed. If the score is less than 0, it will display 0.
        game_font (pygame.font.Font): The font used to render the score text.
        screen (pygame.Surface): The game screen surface where the score will be displayed.

    Returns:
        None
    """
    if score < 0:
        score_surface = game_font.render(str(int(0)), True, (255, 255, 255))
    else:
        score_surface = game_font.render(str(int(score)), True, (255, 255, 255))
    score_rect = score_surface.get_rect(center=(288, 100))
    screen.blit(score_surface, score_rect)


def run(weight_list=None, slow=False, fps=60):
    """
    Runs the Flappy Bird game with optional neural network weights and game speed settings.
    Parameters:
    weight_list (list, optional): A list of weights for the neural network controlling the bird. Defaults to None.
    slow (bool, optional): If True, the game runs at a slower speed for easier observation. Defaults to False.
    fps (int, optional): Frames per second for the game. Defaults to 60.
    Returns:
    list: A list of scores for each bird after the game ends.
    """
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
    """
    Initializes the game by setting up the display, loading assets, and configuring timers.
    Returns:
        screen (pygame.Surface): The main display surface.
        game_font (pygame.font.Font): The font used for displaying text in the game.
        floor_surface (pygame.Surface): The surface for the floor image.
        bg_surface (pygame.Surface): The surface for the background image.
        BIRDFLAP (int): The custom event ID for bird flap events.
        pipe_surface (pygame.Surface): The surface for the pipe image.
    """
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
    """
    Callback function to be called after each generation in the genetic algorithm.

    Args:
        ga_instance: An instance of the genetic algorithm containing information about the current state of the algorithm.

    Prints:
        The current generation number and the score of the best solution found so far.
    """
    print(
        f"Generation: {ga_instance.generations_completed} ------------------ Score: {ga_instance.best_solution()[1]}"
    )


# Define the fitness function


def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates the fitness of a given solution in the genetic algorithm.
    Parameters:
    ga_instance (GAInstance): The instance of the genetic algorithm.
    solution (list): The solution to be evaluated, represented as a list of weights.
    solution_idx (int): The index of the solution in the population.
    Returns:
    float: The fitness value of the solution, which is the negative of the game scores.
    """
    # Convert the solution to numpy array
    weights = np.array(solution)
    # Run the game and get the scores
    scores = run(weights)

    # Return the negative scores as the fitness values
    return scores


def train_from_scratch():
    """
    Trains a neural network from scratch using a genetic algorithm.
    Prompts the user to input the number of generations and population size for the genetic algorithm.
    If the inputs are not valid integers, default values are used.
    The genetic algorithm is configured with the specified parameters and runs to optimize the neural network weights.
    The best solution found by the genetic algorithm is saved to a file and the fitness graph is plotted.
    Parameters:
    None
    Returns:
    None
    """
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
    """
    Trains a neural network using genetic algorithms with the last saved results.
    This function loads the best bird's weights from a file, prompts the user for the number of generations
    and population size, and then trains the neural network using a genetic algorithm. The training process
    uses the previously saved best bird's weights as the initial population.
    The function performs the following steps:
    1. Loads the best bird's weights from "best_birds.npy".
    2. Prompts the user to input the number of generations and population size.
    3. Validates the user inputs and uses default values if the inputs are invalid.
    4. Initializes the population with the best bird's weights.
    5. Configures and runs the genetic algorithm.
    6. Saves the best solution's weights back to "best_birds.npy".
    7. Attempts to plot the fitness graph of the genetic algorithm.
    Note:
        - The function uses global constants: GENERATIONS, POPULATION_SIZE, NEURAL_NETWORK_SHAPE,
          callback_generation, and fitness_func.
        - The function requires the `pygad` and `numpy` libraries.
    Raises:
        Exception: If the user inputs for generations or population size are not valid integers.
        Exception: If there is an error plotting the fitness graph.
    """
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
    """
    Runs the best result of the Flappy Bird game using a neural network and genetic algorithm.
    Prompts the user to enter the desired maximum frames per second (FPS) for the game.
    If the input is not a valid integer, it defaults to 60 FPS.
    Loads the best performing birds from a file named 'best_birds.npy' and runs the game
    with these birds.
    Prints the score achieved by the best birds.
    Returns:
        None
    """
    # Get the desired max FPS
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

    # Prompt the user to select the run mode
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

    run_mode = answers["run_mode"]

    # Run the selected mode
    if run_mode == "TRAIN_FROM_SCRATCH":
        train_from_scratch()
    elif run_mode == "TRAIN_WITH_LAST_RESULTS":
        train_with_last_results()
    elif run_mode == "RUN_BEST_RESULT":
        run_best_result()
    else:
        print("Invalid run mode")
        exit(1)
