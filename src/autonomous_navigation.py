import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

# Grid size and initialization
GRID_SIZE = (50, 50)
occupancy_grid = np.zeros(GRID_SIZE)

# Robot and goal setup
robot_pos = [10, 10]
goal_pos = (40, 40)
obstacles = {(25, y) for y in range(10, 40)}  # Wall

# Mark obstacles as fully occupied
for (ox, oy) in obstacles:
    occupancy_grid[ox, oy] = 1.0

# LIDAR simulation
def update_with_lidar(grid, robot_pos, obstacles, lidar_range=10):
    for angle in np.linspace(0, 2 * np.pi, 36):
        for r in range(1, lidar_range):
            x = int(robot_pos[0] + r * np.cos(angle))
            y = int(robot_pos[1] + r * np.sin(angle))
            if 0 <= x < GRID_SIZE[0] and 0 <= y < GRID_SIZE[1]:
                if (x, y) in obstacles:
                    grid[x, y] = 1.0
                    break
                else:
                    grid[x, y] = max(0, grid[x, y] - 0.05)
    return grid

# A* pathfinding
def a_star(start, goal, grid):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < GRID_SIZE[0] and
                0 <= neighbor[1] < GRID_SIZE[1] and
                grid[neighbor] < 0.5):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
    return []

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(occupancy_grid.T, origin='lower', cmap='gray_r')
robot_dot, = ax.plot([], [], 'bo', label='Robot')
goal_dot, = ax.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal')
path_line, = ax.plot([], [], 'g-', linewidth=2, label='Path')
obstacle_pts = [ax.plot(ox, oy, 'ks')[0] for ox, oy in obstacles]
plt.legend()

# Animate
def init():
    img.set_data(occupancy_grid.T)
    robot_dot.set_data([], [])
    path_line.set_data([], [])
    return [img, robot_dot, goal_dot, path_line]

def update(frame):
    global robot_pos, occupancy_grid
    occupancy_grid = update_with_lidar(occupancy_grid, robot_pos, obstacles)
    path = a_star(tuple(robot_pos), goal_pos, occupancy_grid)
    
    if path and len(path) > 1:
        # Move one step along the path
        robot_pos[0], robot_pos[1] = path[1]
        px, py = zip(*path)
        path_line.set_data(px, py)
    else:
        path_line.set_data([], [])

    robot_dot.set_data(robot_pos[0], robot_pos[1])
    img.set_data(occupancy_grid.T)
    return [img, robot_dot, path_line]

ani = animation.FuncAnimation(fig, update, init_func=init, frames=500, interval=100, blit=True)
plt.title("Live Autonomous Navigation")
plt.grid(True)
plt.show()
