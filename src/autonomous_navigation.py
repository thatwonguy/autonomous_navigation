import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import heapq

# Grid setup
GRID_SIZE = (50, 50)  # (columns, rows)
free, obstacle = 0, 1
full_map = np.zeros(GRID_SIZE, dtype=int)

# Define more elaborate static obstacles (warehouse-like shelves)
def create_obstacles():
    obs = set()
    # Horizontal shelves at y = 10,20,30,40 with aisle gaps every 10 cols
    for y in (10, 20, 30, 40):
        for x in range(5, 45):
            if x % 10 not in (0, 5):  # leave gaps at columns 5 and 15,25...
                obs.add((x, y))
    # Vertical shelves at x = 15,30 with aisle gaps every 10 rows
    for x in (15, 30):
        for y in range(5, 45):
            if y % 10 not in (0, 5):
                obs.add((x, y))
    # Additional L-shaped barrier for complexity
    for x in range(20, 28): obs.add((x, 5))
    for y in range(5, 15): obs.add((27, y))
    return obs

obstacles = create_obstacles()
for ox, oy in obstacles:
    full_map[ox, oy] = obstacle

# A* pathfinding
def a_star(start, goal, grid):
    rows, cols = grid.shape
    def heuristic(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        if current in visited:
            continue
        visited.add(current)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == free:
                tentative_g = g + np.hypot(dx, dy)
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
    return []

# Initialize robot and goal
robot_pos = (5, 5)
goal_pos = (45, 45)
current_path = []

# GUI setup
root = tk.Tk()
root.title("Autonomous Navigation")
messagebox.showinfo("Instructions",
    "🔵 Robot (blue)\n🔴 Goal (red)\n🟫 Obstacles (black)\nClick 'Change Goal' to choose new destination")

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
# Draw obstacles
ox, oy = zip(*obstacles)
ax.plot(ox, oy, 'ks', markersize=5, label='Obstacle')
# Initial plot elements
robot_dot, = ax.plot([], [], 'bo', label='Robot')
goal_dot, = ax.plot([], [], 'ro', label='Goal')
path_line, = ax.plot([], [], 'g-', linewidth=2, label='Path')
# Fix axis limits so goal stays in view
ax.set_xlim(-1, GRID_SIZE[0])
ax.set_ylim(-1, GRID_SIZE[1])
ax.set_aspect('equal')
plt.legend()

# Change goal and replanning
def change_goal():
    global goal_pos, current_path
    while True:
        x, y = np.random.randint(0, GRID_SIZE[0]), np.random.randint(0, GRID_SIZE[1])
        if full_map[x, y] == free:
            goal_pos = (x, y)
            break
    goal_dot.set_data([goal_pos[0]], [goal_pos[1]])
    # recompute path
    current_path = a_star(robot_pos, goal_pos, full_map)
    if not current_path:
        print("⚠️ No path found to new goal")
    else:
        print(f"🔄 Path length: {len(current_path)}")
    # redraw canvas
    ax.figure.canvas.draw_idle()

tk.Button(root, text='Change Goal', command=change_goal).pack(pady=5)

# Animation initialization
def init():
    global current_path
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    goal_dot.set_data([goal_pos[0]], [goal_pos[1]])
    current_path = a_star(robot_pos, goal_pos, full_map)
    path_line.set_data([], [])
    return [robot_dot, goal_dot, path_line]

# Frame update: step along A* path

def update(frame):
    global robot_pos, current_path
    if current_path:
        robot_pos = current_path.pop(0)
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    if current_path:
        xs, ys = zip(*([robot_pos] + current_path))
        path_line.set_data(xs, ys)
    return [robot_dot, path_line]

ani = animation.FuncAnimation(
    fig, update, init_func=init, frames=1000, interval=100, blit=False
)
root.mainloop()
