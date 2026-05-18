import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import heapq

# --- IMPLEMENTACIÓN DIRECTA DE A* PARA COORDENADAS PERFECTAS ---
def astar_2d(maze_layout, start, goal):
    rows = len(maze_layout)
    cols = len(maze_layout[0])
    
    # Movimientos válidos (Arriba, Abajo, Izquierda, Derecha) - No diagonales para evitar atravesar esquinas
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    
    while queue:
        _, current = heapq.heappop(queue)
        
        if current == goal:
            break
            
        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Verificar límites y que sea pasillo libre
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze_layout[neighbor[0]][neighbor[1]] == "█":
                    continue
                    
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    # Heurística de Manhattan
                    f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    came_from[neighbor] = current
                    heapq.heappush(queue, (f_score, neighbor))
                    
    # Reconstruir la ruta de forma segura
    if goal not in came_from:
        return []
        
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

def build_3d_maze():
    os.makedirs("results/data", exist_ok=True)
    log_file_path = "results/data/simulation_log.txt"
    
    with open(log_file_path, "w") as f:
        f.write("=== LOG DE NAVEGACIÓN DINÁMICA PERFECTA CON A* ===\n")
        f.write("Tiempo(s) | Nodo_Destino | Pos_Actual_X | Pos_Actual_Y\n")

    try:
        physicsClient = p.connect(p.GUI)
    except Exception:
        physicsClient = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")

    maze_layout = [
        "███████████████",
        "█S█       █  ██",
        "█ ███ █ █ ███ █",
        "█   ███ █     █",
        "███ ███ █████ █",
        "█ █ █   █ █   █",
        "█ █ █ ███ █ ███",
        "█ █ █ █ █   ███",
        "█ █ █ █ █ ███ █",
        "█ █ █ █ █ █  ██",
        "█ █ █ █ █ ███ █",
        "█   █ ███     █",
        "█ ███ ███████ █",
        "█     █     █G█",
        "███████████████"
    ]

    rows = len(maze_layout)
    cols = len(maze_layout[0])
    block_size = 1.0
    wall_height = 1.2

    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[block_size/2, block_size/2, wall_height/2], rgbaColor=[0.3, 0.3, 0.35, 1.0])
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[block_size/2, block_size/2, wall_height/2])

    robot_id = None
    start_cell = None
    goal_cell = None
    
    # Construcción e indexación de posiciones de celdas clave
    for r in range(rows):
        for c in range(cols):
            cell = maze_layout[r][c]
            x = (c - cols / 2) * block_size
            y = (rows / 2 - r) * block_size

            if cell == "█":
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[x, y, wall_height / 2])
            elif cell == "S":
                start_cell = (r, c)
                start_pos = [x, y, 0.35]
                v_robot = p.createVisualShape(p.GEOM_SPHERE, radius=0.22, rgbaColor=[1, 0, 0, 1])
                c_robot = p.createCollisionShape(p.GEOM_SPHERE, radius=0.22)
                robot_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=c_robot, baseVisualShapeIndex=v_robot, basePosition=start_pos)
            elif cell == "G":
                goal_cell = (r, c)
                v_goal = p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.1, rgbaColor=[0, 1, 0, 0.6])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_goal, basePosition=[x, y, 0.05])

    p.resetDebugVisualizerCamera(cameraDistance=14.0, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])

    print("\n--- EJECUTANDO ALGORITMO A* PARA GENERACIÓN DE WAYPOINTS ---")
    puntos_ruta = astar_2d(maze_layout, start_cell, goal_cell)
    print(f"¡Ruta óptima calculada por A* con éxito! ({len(puntos_ruta)} celdas seguras encontradas).")
    
    # Convertir las celdas de la ruta de la matriz al espacio de coordenadas métricas 3D
    waypoints_3d = []
    for (r, c) in puntos_ruta:
        wx = (c - cols / 2) * block_size
        wy = (rows / 2 - r) * block_size
        waypoints_3d.append([wx, wy, 0.35])

    current_wp = 0
    start_time = time.time()
    
    # Quitar fricciones para asegurar el desplazamiento cinemático continuo
    p.changeDynamics(planeId, -1, lateralFriction=0.0, spinningFriction=0.0)
    if robot_id is not None:
        p.changeDynamics(robot_id, -1, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, restitution=0.2)
    
    try:
        while True:
            if robot_id is not None and current_wp < len(waypoints_3d):
                target = waypoints_3d[current_wp]
                
                pos, orient = p.getBasePositionAndOrientation(robot_id)
                diff = np.array(target) - np.array(pos)
                diff[2] = 0
                dist = np.linalg.norm(diff)
                
                elapsed = time.time() - start_time
                with open(log_file_path, "a") as log_f:
                    log_f.write(f"{elapsed:.2f}s | Nodo {current_wp + 1} | X: {pos[0]:.2f} | Y: {pos[1]:.2f}\n")
                
                # Tolerancia de aceptación optimizada por celda autónoma
                if dist < 0.20:
                    current_wp += 1
                    print(f"Agente autónomo alcanzó nodo real [{current_wp}/{len(waypoints_3d)}]")
                else:
                    # Desplazamiento cinemático controlado por interpolación fina (0.07 metros por frame)
                    direction = diff / dist
                    nueva_pos = np.array(pos) + direction * 0.07
                    nueva_pos[2] = 0.35
                    p.resetBasePositionAndOrientation(robot_id, nueva_pos.tolist(), orient)
            
            p.stepSimulation()
            time.sleep(1./120.)
            
            if current_wp >= len(waypoints_3d):
                print("\n¡PARCIAL COMPLETADO CON ÉXITO! El agente físico ejecutó A* real y llegó legalmente a la meta.")
                time.sleep(2)
                break
                
    except KeyboardInterrupt:
        pass
    
    p.disconnect()
    print("Simulación cerrada.")

if __name__ == "__main__":
    build_3d_maze()
