import heapq
import time
import pybullet as p
import pybullet_data
import math

# ==========================================
# LABERINTO REAL 5x5 (Con bordes externos)
# ==========================================
laberinto_fijo = [
    "███████",
    "█S    █",
    "█████ █",
    "█ █   █",
    "█   ███",
    "█ █  G█",
    "███████"
]

INICIO = (1, 1)  # Coordenada de 'S'
META   = (5, 5)  # Coordenada de 'G'

# ==========================================
# ALGORITMO A*
# ==========================================
def astar_2d(maze, start, goal):
    rows = len(maze)
    cols = len(maze[0])
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    queue = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}

    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        for dr, dc in dirs:
            nb = (current[0]+dr, current[1]+dc)
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols:
                if maze[nb[0]][nb[1]] == '█':
                    continue
                tg = g_score[current] + 1
                if nb not in g_score or tg < g_score[nb]:
                    g_score[nb] = tg
                    f = tg + abs(nb[0]-goal[0]) + abs(nb[1]-goal[1])
                    came_from[nb] = current
                    heapq.heappush(queue, (f, nb))

    if goal not in came_from:
        return []
    path, curr = [], goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

# ==========================================
# CONVERSIÓN CELDA → POSICIÓN 3D
# ==========================================
CELL  = 1.0  
WALL_H = 0.6
ROBOT_H = 0.18

def cell_to_xyz(r, c):
    return (c * CELL, -r * CELL, ROBOT_H)

# ==========================================
# CONSTRUIR MUNDO EN PYBULLET
# ==========================================
def build_world(maze, path):
    rows = len(maze)
    cols = len(maze[0])
    path_set = set(path)

    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cols*CELL/2, rows*CELL/2, 0.01])
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cols*CELL/2, rows*CELL/2, 0.01], rgbaColor=[0.85, 0.85, 0.85, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_col, baseVisualShapeIndex=floor_vis, basePosition=[cols*CELL/2 - CELL/2, -rows*CELL/2 + CELL/2, -0.01])

    wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CELL/2, CELL/2, WALL_H/2])

    for r in range(rows):
        for c in range(cols):
            cell = maze[r][c]
            x, y, _ = cell_to_xyz(r, c)

            if cell == '█':
                wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CELL/2, CELL/2, WALL_H/2], rgbaColor=[0.25, 0.25, 0.30, 1])
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis, basePosition=[x, y, WALL_H/2])
            elif cell == 'S' or (r, c) == INICIO:
                sv = p.createVisualShape(p.GEOM_CYLINDER, radius=CELL*0.38, length=0.04, rgbaColor=[0.1, 0.85, 0.1, 0.8])
                p.createMultiBody(baseVisualShapeIndex=sv, basePosition=[x, y, 0.02])
            elif cell == 'G' or (r, c) == META:
                gv = p.createVisualShape(p.GEOM_CYLINDER, radius=CELL*0.38, length=0.04, rgbaColor=[1.0, 0.15, 0.15, 0.8])
                p.createMultiBody(baseVisualShapeIndex=gv, basePosition=[x, y, 0.02])

            if (r, c) in path_set and cell not in ('█',):
                pv = p.createVisualShape(p.GEOM_BOX, halfExtents=[CELL*0.25, CELL*0.25, 0.01], rgbaColor=[1.0, 0.75, 0.0, 0.6])
                p.createMultiBody(baseVisualShapeIndex=pv, basePosition=[x, y, 0.015])

# ==========================================
# CREAR ROBOT
# ==========================================
def create_robot():
    body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.18, 0.10, 0.06])
    body_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.10, 0.06], rgbaColor=[0.1, 0.4, 0.9, 1])
    head_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.07, 0.07, 0.05], rgbaColor=[0.05, 0.25, 0.7, 1])

    sx, sy, sz = cell_to_xyz(*INICIO)
    robot_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=body_col, baseVisualShapeIndex=body_vis, basePosition=[sx, sy, sz])
    p.createMultiBody(baseVisualShapeIndex=head_vis, basePosition=[sx + 0.20, sy, sz + 0.05])
    return robot_id

# ==========================================
# MOVER ROBOT
# ==========================================
def move_robot(robot_id, path):
    STEPS = 50  
    SPEED = 1/120  

    for i in range(1, len(path)):
        r_prev, c_prev = path[i-1]
        r_next, c_next = path[i]

        x0, y0, z0 = cell_to_xyz(r_prev, c_prev)
        x1, y1, z1 = cell_to_xyz(r_next, c_next)

        dx = x1 - x0
        dy = y1 - y0
        yaw = math.atan2(dy, dx)
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        for step in range(STEPS):
            t = step / STEPS
            t_smooth = t * t * (3 - 2 * t)
            xi = x0 + t_smooth * (x1 - x0)
            yi = y0 + t_smooth * (y1 - y0)
            bounce = abs(math.sin(step * math.pi / STEPS)) * 0.02
            zi = z0 + bounce

            p.resetBasePositionAndOrientation(robot_id, [xi, yi, zi], quat)
            p.stepSimulation()
            time.sleep(SPEED)

        xv, yv, _ = cell_to_xyz(r_next, c_next)
        trail_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 1, 0.9])
        p.createMultiBody(baseVisualShapeIndex=trail_vis, basePosition=[xv, yv, 0.08])

    print("\n[3D] ¡Meta alcanzada en la simulación!")

# ==========================================
# MAIN
# ==========================================
def main():
    ruta = astar_2d(laberinto_fijo, INICIO, META)
    if not ruta: 
        print("[ERROR] No se pudo calcular una ruta.")
        return

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    rows, cols = len(laberinto_fijo), len(laberinto_fijo[0])
    p.resetDebugVisualizerCamera(cameraDistance=6.0, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[cols*CELL/2, -rows*CELL/2, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    build_world(laberinto_fijo, ruta)
    robot_id = create_robot()
    time.sleep(1)
    move_robot(robot_id, ruta)

    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1/60)
    except: pass
    p.disconnect()

if __name__ == '__main__':
    main()
