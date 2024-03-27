import sys, pygame, random
import numpy as np
import joblib


model = joblib.load("trained_model.pkl")


class Birds:

    def __init__(self):
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
                return True
            if bird_rect.top <= -100 or bird_rect.bottom >= 900:
                return True
    return False


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


def run():
    global model
    Bird = Birds()
    floor_x_pos = 0
    pipe_list = []
    game_active = True
    score = 0

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

        # GENERATING BIRD DATA
        pipe_parameters = np.append(
            pipe_parameters, [Bird.rect.centery, Bird.rect.centerx, Bird.movement]
        )

        result = model.predict(pipe_parameters.reshape(1, -1))
        print(result)
        if result > 0.5:
            Bird.movement = 0
            Bird.movement -= 10

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == BIRDFLAP:
                if Bird.alive:
                    if Bird.index < 2:
                        Bird.index += 1
                    else:
                        Bird.index = 0
                    Bird.surface, Bird.rect = bird_animation(
                        Bird.frames, Bird.index, Bird.rect
                    )
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    Bird.movement = 0
                    Bird.movement -= 10
                    # flap_sound.play()

        screen.blit(bg_surface, (0, 0))
        if (pipe_list.__len__() > 0 and pipe_list[-1][0].centerx < 200) or (
            pipe_list.__len__() == 0
        ):
            if pipe_list.__len__() >= 3:
                pipe_list.pop(0)
                score += 1
            pipe_list.extend(create_pipe())

        #

        # FLOOR
        if game_active:
            Bird.movement += gravity
            Bird.rect.centery += Bird.movement
            rotated_bird = rotate_bird(Bird.surface, Bird.movement)
            screen.blit(rotated_bird, Bird.rect)
            collision = check_collision(pipe_list, Bird.rect)
            if collision:
                Bird.alive = False
                game_active = False
            floor_x_pos -= 4
            # PIPES
            pipe_list = move_pipes(pipe_list)
            draw_pipes(pipe_list)
            score_display(score)
        draw_floor(floor_x_pos)
        if floor_x_pos <= -576:
            floor_x_pos = 0

        Clock.tick(60)
        pygame.display.update()


# pygame.mixer.pre_init(frequency=44100, size=32, channels=1, buffer=512)
pygame.init()
Clock = pygame.time.Clock()

screen = pygame.display.set_mode((576, 1024))
game_font = pygame.font.Font("FlappyBirdy.ttf", 72)


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


run()
