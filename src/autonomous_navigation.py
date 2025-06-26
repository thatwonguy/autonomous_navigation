import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Grid setup
GRID_SIZE = (50, 50)
full_map = np.zeros(GRID_SIZE)
known_map = np.full(GRID_SIZE, 0.5)

# Start and goal
robot_pos = [10, 10]
goal_pos = [40, 40]
robot_history = []

# Add complex static obstacles
obstacles = set()
# Vertical wall with 3 gaps
for y in range(10, 40):
    if y not in (18, 25, 32):
        obstacles.add((25, y))
# Horizontal wall
for x in range(5, 45):
    if x not in (12, 26, 38):
        obstacles.add((x, 15))
# L-shaped barrier
for x in range(30, 35):
    obstacles.add((x, 35))
for y in range(35, 45):
    obstacles.add((30, y))

for ox, oy in obstacles:
    full_map[ox, oy] = 1.0

# GUI
root = tk.Tk()
root.title("Autonomous Navigation")

messagebox.showinfo("Instructions",
    "🔵 Robot (blue)\n🔴 Goal (red)\n🟢 Path (green)\nClick 'Change Goal' to relocate goal")

fig, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

img = ax.imshow(known_map.T, origin='lower', cmap='gray_r')
robot_dot, = ax.plot([], [], 'bo', label='Robot')
goal_dot, = ax.plot([goal_pos[0]], [goal_pos[1]], 'ro', label='Goal')
path_line, = ax.plot([], [], 'g-', linewidth=2, label='Path')

# After defining `obstacles`
obs_x, obs_y = zip(*obstacles)
ax.plot(obs_x, obs_y, 'ks', markersize=5, label='Obstacle')  # 'ks' = black squares

plt.legend()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

# Change goal location
def change_goal():
    global goal_pos
    while True:
        goal_pos = [np.random.randint(0, GRID_SIZE[0]), np.random.randint(0, GRID_SIZE[1])]
        if full_map[goal_pos[0], goal_pos[1]] == 0:
            break
    goal_dot.set_data([goal_pos[0]], [goal_pos[1]])
    print(f"🎯 New goal: {goal_pos}")

tk.Button(btn_frame, text="Change Goal", command=change_goal).pack()

# Find valid 8-direction neighbors
def get_neighbors(pos):
    x, y = pos
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
            if full_map[nx, ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

# Choose next move greedily toward goal
def choose_step():
    current = tuple(robot_pos)
    neighbors = get_neighbors(current)
    if not neighbors:
        return current
    # Sort neighbors by distance to goal (greedy)
    neighbors.sort(key=lambda n: np.linalg.norm(np.array(n) - np.array(goal_pos)))
    for next_pos in neighbors:
        if next_pos not in robot_history:  # Avoid cycles
            return next_pos
    return current  # No good move found

# Animation init
def init():
    img.set_data(known_map.T)
    robot_dot.set_data([], [])
    path_line.set_data([], [])
    return [img, robot_dot, path_line]

# Frame update
def update(frame):
    if tuple(robot_pos) == tuple(goal_pos):
        return [img, robot_dot, path_line]

    next_pos = choose_step()
    if next_pos != tuple(robot_pos):
        robot_pos[0], robot_pos[1] = next_pos
        robot_history.append(tuple(robot_pos))

    hx, hy = zip(*robot_history)
    path_line.set_data(hx, hy)
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    return [img, robot_dot, path_line]

ani = animation.FuncAnimation(fig, update, init_func=init, frames=1000, interval=100, blit=False)
root.mainloop()
