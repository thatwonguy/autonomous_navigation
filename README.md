# autonomous_navigation
This is a live simulation of autonomous robot navigation carried out programmatically.
# Autonomous Navigation (Live Simulation)

This is a live simulation of autonomous robot navigation using:

- Probabilistic occupancy grid
- Simulated LIDAR
- A* pathfinding
- Live GUI animation with `matplotlib`

## Setup

1. From the root directory of the repo type the following, ideally in a virtual env:

```bash
pip install -r requirements.txt
```

2. Then run the following after changing directory into the `src` folder where the main script is located.

```bash
python autonomous_navigation.py
```
---
### The simulation opens a window showing:

- Robot (blue dot)

- Goal (red dot)

- Obstacles (black)

- Live path planning (green path)
