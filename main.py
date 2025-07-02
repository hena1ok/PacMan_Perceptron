import pygame
import sys
import random
import math
import numpy as np

pygame.init()

# Colors
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 650
CELL_SIZE = 40

# Grid dimensions
GRID_WIDTH = 15
GRID_HEIGHT = 15

# Game states
PLAYING = 0
GAME_OVER = 1
# Global game state
game_state = PLAYING

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pac-Man")

# Font for score
font = pygame.font.Font(None, 36)

# Game grid
grid = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Pac-Man
pacman = {
    'x': 1,
    'y': 1,
    'direction': 3,  # 0: right, 1: down, 2: left, 3: up
    'mouth_open': False
}

# Ghosts
ghosts = [
    {'x': 1, 'y': 13, 'color': RED},
    {'x': 13, 'y': 1, 'color': PINK},
    {'x': 13, 'y': 13, 'color': CYAN},
    {'x': 11, 'y': 11, 'color': ORANGE}
]

# Score
score = 0

# Game Loop
clock = pygame.time.Clock()
running = True

# Movement delays
pacman_move_delay = 150  # milliseconds
ghost_move_delay = 300
mouth_anim_delay = 600
# Timing variables
last_pacman_move_time = 0
last_ghost_move_time = 0
last_mouth_anim_time = 0

def check_valid_move(direction):
    dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
    new_x = pacman['x'] + dx
    new_y = pacman['y'] + dy
    return grid[new_y][new_x] != 1

def get_valid_directions():
    valid = []
    for direction in range(4):
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
        new_x = pacman['x'] + dx
        new_y = pacman['y'] + dy
        if grid[new_y][new_x] != 1:
            valid.append(direction)
    return valid


class Perceptron:
      def __init__(self, input_size, output_size, learning_rate=0.1):
          self.weights = np.random.randn(input_size, output_size) * 0.1
          self.bias = np.zeros(output_size)
          self.learning_rate = learning_rate
          self.epsilon = 0.03  # Lower exploration rate
        
      def get_action(self, state):
          output = self.forward(state)
          valid_moves = get_valid_directions()
        
          move_scores = []
          for move in get_valid_directions():
              dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][move]
              new_x = pacman['x'] + dx
              new_y = pacman['y'] + dy
            
              score = output[move]
            
              # Immediate pellet reward
              if grid[new_y][new_x] == 0:
                  score += 1.0
            
              # Path value calculation
              path_value = 0
              test_x, test_y = new_x, new_y
              for _ in range(4):  # Look ahead 4 steps
                  if (0 <= test_x + dx < GRID_WIDTH and 
                      0 <= test_y + dy < GRID_HEIGHT and 
                      grid[test_y+dy][test_x+dx] != 1):
                      test_x += dx
                      test_y += dy
                      if grid[test_y][test_x] == 0:
                          path_value += 0.25  # Diminishing returns for distant pellets
              score += path_value
            
              # Ghost avoidance
              for ghost in ghosts:
                  ghost_dist = math.sqrt((ghost['x'] - new_x)**2 + (ghost['y'] - new_y)**2)
                  if ghost_dist < 3:
                      score -= (3 - ghost_dist) * 1.5
            
              # Prevent backtracking unless necessary
              if (dx, dy) == ((-1 if pacman['direction'] == 0 else 1 if pacman['direction'] == 2 else 0),
                           (-1 if pacman['direction'] == 1 else 1 if pacman['direction'] == 3 else 0)):
                  score -= 0.3
            
              move_scores.append((move, score))
        
          if random.random() < self.epsilon:
              return random.choice(valid_moves)
        
          return max(move_scores, key=lambda x: x[1])[0] if move_scores else pacman['direction']

      def forward(self, inputs):
          self.inputs = np.array(inputs)
          self.output = self.sigmoid(np.dot(self.inputs, self.weights) + self.bias)
          return self.output

      def sigmoid(self, x):
          return 1 / (1 + np.exp(-x))

      def backward(self, target):
          error = target - self.output
          gradient = error * self.output * (1 - self.output)
          self.weights += self.learning_rate * np.outer(self.inputs, gradient)
          self.bias += self.learning_rate * gradient

  # Initialize with expanded state size
perceptron = Perceptron(input_size=21, output_size=4)  # 9 ghost features + 12 cluster features
def get_game_state():
    # Get distances to all ghosts
    ghost_distances = []
    for ghost in ghosts:
        dx = ghost['x'] - pacman['x']
        dy = ghost['y'] - pacman['y']
        distance = math.sqrt(dx**2 + dy**2)
        ghost_distances.append(distance)
    
    # Get nearest ghost distance
    nearest_ghost = min(ghost_distances)
    
    # Check walls in all 4 directions (right, down, left, up)
    walls = []
    for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
        new_x = pacman['x'] + dx
        new_y = pacman['y'] + dy
        walls.append(1.0 if grid[new_y][new_x] == 1 else 0.0)
            
    # Normalize values
    normalized_ghost_dist = nearest_ghost / math.sqrt(GRID_WIDTH**2 + GRID_HEIGHT**2)
    
    # Return state including wall information
    return [normalized_ghost_dist, score/1000] + walls

# Update perceptron initialization to handle more inputs
perceptron = Perceptron(input_size=6, output_size=4)  # 6 inputs: ghost distance, score, 4 wall directions

def move_pacman():
    global score
    dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][pacman['direction']]
    new_x, new_y = pacman['x'] + dx, pacman['y'] + dy
    if grid[new_y][new_x] != 1:
        pacman['x'], pacman['y'] = new_x, new_y
        if grid[new_y][new_x] == 0:
            grid[new_y][new_x] = 2  # Mark as eaten
            score += 10
            return 1  # Positive reward for eating
    return 0  # No reward

def move_ghost(ghost):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    random.shuffle(directions)
    for dx, dy in directions:
        new_x, new_y = ghost['x'] + dx, ghost['y'] + dy
        if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and grid[new_y][new_x] != 1:
            ghost['x'], ghost['y'] = new_x, new_y
            break

def draw_pacman():
    x = pacman['x'] * CELL_SIZE + CELL_SIZE // 2
    y = pacman['y'] * CELL_SIZE + CELL_SIZE // 2 + 50

    # Mouth opening angle varies between 0 (fully closed) and 45 degrees (fully open)
    mouth_opening = 45 if pacman['mouth_open'] else 0

    # Draw Pac-Man as a circle
    pygame.draw.circle(screen, YELLOW, (x, y), CELL_SIZE // 2)

    # Calculate the angles for the mouth based on direction
    if pacman['direction'] == 0:  # Right
        start_angle = 360 - mouth_opening / 2
        end_angle = mouth_opening / 2
    elif pacman['direction'] == 3:  # Down
        start_angle = 90 - mouth_opening / 2
        end_angle = 90 + mouth_opening / 2
    elif pacman['direction'] == 2:  # Left
        start_angle = 180 - mouth_opening / 2
        end_angle = 180 + mouth_opening / 2
    else:  # Up
        start_angle = 270 - mouth_opening / 2
        end_angle = 270 + mouth_opening / 2

    # Draw the mouth using a pie shape (filled arc)
    pygame.draw.arc(screen, BLACK, 
                    (x - CELL_SIZE // 2, y - CELL_SIZE // 2, CELL_SIZE, CELL_SIZE), 
                    math.radians(start_angle), math.radians(end_angle), CELL_SIZE // 2)
    
    # Draw a line from the center to create the "slice" effect
    mouth_line_end_x = x + math.cos(math.radians(start_angle)) * CELL_SIZE // 2
    mouth_line_end_y = y - math.sin(math.radians(start_angle)) * CELL_SIZE // 2
    pygame.draw.line(screen, BLACK, (x, y), (mouth_line_end_x, mouth_line_end_y), 2)

    mouth_line_end_x = x + math.cos(math.radians(end_angle)) * CELL_SIZE // 2
    mouth_line_end_y = y - math.sin(math.radians(end_angle)) * CELL_SIZE // 2
    pygame.draw.line(screen, BLACK, (x, y), (mouth_line_end_x, mouth_line_end_y), 2)

def draw_ghost(ghost):
    x = ghost['x'] * CELL_SIZE + CELL_SIZE // 2
    y = ghost['y'] * CELL_SIZE + CELL_SIZE // 2 + 50
    pygame.draw.circle(screen, ghost['color'], (x, y), CELL_SIZE // 2)

def reset_game():
    global pacman, ghosts, score, grid, game_state
    pacman = {
        'x': 1,
        'y': 1,
        'direction': 3,
        'mouth_open': False
    }
    ghosts = [
        {'x': 1, 'y': 13, 'color': RED},
        {'x': 13, 'y': 1, 'color': PINK},
        {'x': 13, 'y': 13, 'color': CYAN},
        {'x': 11, 'y': 11, 'color': ORANGE}
    ]
    score = 0
    grid = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    game_state = PLAYING

def draw_game_over():
    screen.fill(BLACK)
    game_over_font = pygame.font.Font(None, 64)
    score_font = pygame.font.Font(None, 48)
    restart_font = pygame.font.Font(None, 36)

    game_over_text = game_over_font.render("GAME OVER", True, RED)
    score_text = score_font.render(f"Score: {score}", True, WHITE)
    restart_text = restart_font.render("Press SPACE to restart", True, YELLOW)
    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 3))
    screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 2 * SCREEN_HEIGHT // 3))

  # Add this constant with other game states
WINNER = 2

  # Add this function to draw the winning screen
def draw_winner_screen():
      screen.fill(BLACK)
      winner_font = pygame.font.Font(None, 74)
      score_font = pygame.font.Font(None, 48)
      restart_font = pygame.font.Font(None, 36)

      winner_text = winner_font.render("CONGRATULATIONS!", True, YELLOW)
      score_text = score_font.render(f"Final Score: {score}", True, WHITE)
      restart_text = restart_font.render("Press SPACE to play again", True, CYAN)

      screen.blit(winner_text, (SCREEN_WIDTH // 2 - winner_text.get_width() // 2, SCREEN_HEIGHT // 3))
      screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2))
      screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 2 * SCREEN_HEIGHT // 3))

  # Add this check in the main game loop where you handle game logic
def check_win_condition():
      for y in range(GRID_HEIGHT):
          for x in range(GRID_WIDTH):
              if grid[y][x] == 0:  # If there's still a pellet
                  return False
      return True

  # Main game loop
running = True
clock = pygame.time.Clock()

while running:
    
      # Event handling
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              running = False
          elif event.type == pygame.KEYDOWN:
              if event.key == pygame.K_SPACE and (game_state == GAME_OVER or game_state == WINNER):
                  reset_game()

      # Game logic
      current_time = pygame.time.get_ticks()
      if game_state == PLAYING:
          if current_time - last_pacman_move_time > pacman_move_delay:
              state = get_game_state()
              action = perceptron.get_action(state)
            
              if check_valid_move(action):
                  pacman['direction'] = action
                  reward = move_pacman()
                
                  # Calculate additional reward based on wall avoidance
                  valid_moves = get_valid_directions()
                  wall_awareness = len(valid_moves) / 4.0  # Reward for having more options
                  reward += wall_awareness
                
                  target = np.zeros(4)
                  target[action] = reward
                  perceptron.backward(target)
            
              last_pacman_move_time = current_time
          if current_time - last_ghost_move_time > ghost_move_delay:
              for ghost in ghosts:
                  move_ghost(ghost)
              last_ghost_move_time = current_time
        
          if current_time - last_mouth_anim_time > mouth_anim_delay:
              pacman['mouth_open'] = not pacman['mouth_open']
              last_mouth_anim_time = current_time

          # Check collisions between Pac-Man and ghosts
          for ghost in ghosts:
              if pacman['x'] == ghost['x'] and pacman['y'] == ghost['y']:
                  game_state = GAME_OVER

          if check_win_condition():
              game_state = WINNER

      # Drawing
      screen.fill(BLACK)

      if game_state == PLAYING:
          # Draw the grid
          for y in range(GRID_HEIGHT):
              for x in range(GRID_WIDTH):
                  cell = grid[y][x]
                  cell_x = x * CELL_SIZE
                  cell_y = y * CELL_SIZE + 50  # Offset for score area
                  if cell == 1:
                      pygame.draw.rect(screen, BLUE, (cell_x, cell_y, CELL_SIZE, CELL_SIZE))
                  elif cell == 0:
                      pygame.draw.circle(screen, WHITE, (cell_x + CELL_SIZE // 2, cell_y + CELL_SIZE // 2), 5)

          # Draw Pac-Man
          draw_pacman()

          # Draw ghosts
          for ghost in ghosts:
              draw_ghost(ghost)

          # Draw score
          score_text = font.render(f"Score: {score}", True, YELLOW)
          screen.blit(score_text, (10, 10))
      elif game_state == GAME_OVER:
          draw_game_over()
      elif game_state == WINNER:
          draw_winner_screen()

      pygame.display.flip()
      clock.tick(60)

pygame.quit()
sys.exit()
