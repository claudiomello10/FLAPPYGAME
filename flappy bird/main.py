import sys, pygame, random, deap


#PLEU

f = open("players.txt")
players = int(f.read())
print(players)

class Birds:
    def __init__(self):
        self.bird_downflap = pygame.transform.scale2x(pygame.image.load("sprites/bluebird-downflap.png").convert_alpha())
        self.bird_midflap = pygame.transform.scale2x(pygame.image.load("sprites/bluebird-midflap.png").convert_alpha())
        self.bird_upflap = pygame.transform.scale2x(pygame.image.load("sprites/bluebird-upflap.png").convert_alpha())
        self.bird_frames = [bird_downflap,bird_midflap,bird_upflap]
        self.bird_index = 2
        self.bird_surface = bird_frames[bird_index]
        self.bird_rect = bird_surface.get_rect(center = (100,512))
        self.bird_movement = 0



def draw_floor():

    screen.blit(floor_surface,(floor_x_pos,900))
    screen.blit(floor_surface,(floor_x_pos + 576,900))

def create_pipe():
    random_pipe_pos = random.randrange(400,800,100)
    bottom_pipe = pipe_surface.get_rect(midtop = (700,random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom = (700,random_pipe_pos - 300))
    return bottom_pipe, top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 4
    return pipes

def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >=1024:
            screen.blit(pipe_surface,pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe)


def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe) or bird_rect.top <= -100 or bird_rect.bottom >= 900:
            return False
    return True   

def rotate_bird(bird):
    new_bird = pygame.transform.rotozoom(bird,-bird_movement*3,1)
    return new_bird
   
def bird_animation():
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center = (100,bird_rect.centery))
    return new_bird,new_bird_rect

def score_display(game_state):
    if game_state == "main_game":
        if score < 0:
            score_surface = game_font.render(str(int(0)),True,(255,255,255))
        else:
            score_surface = game_font.render(str(int(score)),True,(255,255,255))
        score_rect = score_surface.get_rect(center = (288, 100))
        screen.blit(score_surface,score_rect)
    if game_state == "game_over":
        high_score_surface = game_font.render(str(int(high_score)),True,(255,255,255))
        high_score_rect = high_score_surface.get_rect(center = (288, 850))
        screen.blit(high_score_surface,high_score_rect)
pygame.mixer.pre_init(frequency = 44100, size = 32, channels = 1, buffer = 512)
pygame.init()

screen = pygame.display.set_mode((576,1024))
clock = pygame.time.Clock()
game_font = pygame.font.Font("FlappyBirdy.ttf",72)

# VARIAVEIS 
gravity = 0.25
bird_movement = 0
game_active = True
score = -1
high_score = 0

#SOUND
flap_sound = pygame.mixer.Sound('audio/wing.wav')

floor_surface = pygame.image.load("sprites/base.png").convert()
floor_surface = pygame.transform.scale2x(floor_surface)



bg_surface = pygame.image.load("sprites/background-day.png").convert()
bg_surface = pygame.transform.scale2x(bg_surface)


bird_downflap = pygame.transform.scale2x(pygame.image.load("sprites/bluebird-downflap.png").convert_alpha())
bird_midflap = pygame.transform.scale2x(pygame.image.load("sprites/bluebird-midflap.png").convert_alpha())
bird_upflap = pygame.transform.scale2x(pygame.image.load("sprites/bluebird-upflap.png").convert_alpha())
bird_frames = [bird_downflap,bird_midflap,bird_upflap]
bird_index = 2
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center = (100,512))

BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP,200)

#bird_surface = pygame.image.load("sprites/bluebird-midflap.png").convert_alpha()
#bird_surface = pygame.transform.scale2x(bird_surface)
#bird_rect = bird_surface.get_rect(center = (100,512))


pipe_surface = pygame.image.load("sprites/pipe-green.png").convert()
pipe_surface = pygame.transform.scale2x(pipe_surface)
pipe_list = []
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 1200)
pipe_height = [400,600,800]




floor_x_pos = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird_movement = -10
                flap_sound.play()
                if len(pipe_list) >= 10:
                    for i in range(0,3):
                        pipe_list.pop(i)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active == False:
                game_active = True
                pipe_list.clear()
                bird_rect.center = (100,512)
                bird_movement = 0
                score = -1
        if event.type == BIRDFLAP:
            if bird_index < 2:
                bird_index += 1
            else:
                bird_index = 0
            bird_surface,bird_rect = bird_animation()
        if event.type == SPAWNPIPE:
            pipe_list.extend(create_pipe())
            if game_active == True:
                score += 1
            
    
    
    screen.blit(bg_surface,(0,0))
    if score -1 > high_score:
        high_score = score-1
    if game_active:
        #BIRD
        bird_movement += gravity
        rotated_bird = rotate_bird(bird_surface)
        bird_rect.centery += bird_movement
        screen.blit(rotated_bird, bird_rect)
        game_active = check_collision(pipe_list)


        #PIPES
        pipe_list = move_pipes(pipe_list)
        draw_pipes(pipe_list)

        score_display("main_game")
    else:
        score_display("game_over")

    #FLOOR
    if game_active:
        floor_x_pos -= 4
    draw_floor()
    if floor_x_pos <= -576:
        floor_x_pos = 0

    pygame.display.update()
    clock.tick(120)
