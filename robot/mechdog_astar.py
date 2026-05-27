import heapq
import time

# Importaciones del MechDog (solo en el robot físico)
try:
    import Board
    import ActionGroupControl as AGC
    ROBOT_MODE = True
except ImportError:
    ROBOT_MODE = False
    print("[SIMULACION] Módulos del robot no disponibles — modo consola")

# ==========================================
# LABERINTO FIJO (tú pones los obstáculos)
# ==========================================
# '█' = Obstáculo físico que traerás tú
# ' ' = Pasillo libre
# 'S' = Inicio del robot
# 'G' = Meta
# Cada celda = 1 paso del robot (~20 cm aprox)

laberinto_fijo = [
    "██████",
    "█S   █",
    "█ ██ █",
    "█  █ █",
    "█   G█",
    "██████"
]

INICIO = (1, 1)
META   = (4, 4)

# ==========================================
# TIEMPOS DE CADA ACCIÓN (ajusta en pruebas)
# ==========================================
TIEMPO_PASO    = 1.2   # segundos por celda hacia adelante
TIEMPO_GIRO_90 = 1.0   # segundos para girar 90 grados
TIEMPO_STAND   = 2.0   # espera inicial parándose

# ==========================================
# NOMBRES DE GRUPOS DE ACCIÓN DEL MECHDOG
# ==========================================
# Ajusta estos nombres según tu firmware
ACTION_STAND     = 'stand'
ACTION_FORWARD   = 'move_forward'
ACTION_TURN_LEFT = 'turn_left'
ACTION_TURN_RIGHT= 'turn_right'
ACTION_STOP      = 'stand'

# ==========================================
# A* SOBRE EL LABERINTO
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
# CONVERTIR RUTA EN DIRECCIONES
# ==========================================
# Orientación: 0=Norte(-row), 1=Este(+col), 2=Sur(+row), 3=Oeste(-col)
DIR_VECTOR = {
    0: (-1, 0),  # Norte
    1: (0,  1),  # Este
    2: (1,  0),  # Sur
    3: (0, -1),  # Oeste
}
DIR_NOMBRE = {0: 'NORTE', 1: 'ESTE', 2: 'SUR', 3: 'OESTE'}

def ruta_a_comandos(path):
    """
    Convierte lista de celdas en lista de comandos:
    ('forward', n_pasos) o ('turn_left'/'turn_right', n_giros)
    """
    if len(path) < 2:
        return []

    # Orientación inicial: mirando al Este (ajusta según cómo pongas el robot)
    orientacion = 1  # Este
    comandos = []

    i = 1
    while i < len(path):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]

        # Determinar dirección deseada
        dir_deseada = None
        for d, (vr, vc) in DIR_VECTOR.items():
            if vr == dr and vc == dc:
                dir_deseada = d
                break

        # Calcular giros necesarios
        diff = (dir_deseada - orientacion) % 4
        if diff == 1:
            comandos.append(('turn_right', 1))
        elif diff == 2:
            comandos.append(('turn_right', 2))
        elif diff == 3:
            comandos.append(('turn_left', 1))
        # diff == 0: sin giro

        orientacion = dir_deseada

        # Contar pasos consecutivos en la misma dirección
        pasos = 1
        while i + pasos < len(path):
            dr2 = path[i+pasos][0] - path[i+pasos-1][0]
            dc2 = path[i+pasos][1] - path[i+pasos-1][1]
            if (dr2, dc2) == (dr, dc):
                pasos += 1
            else:
                break

        comandos.append(('forward', pasos))
        i += pasos

    return comandos

# ==========================================
# EJECUTAR COMANDO EN EL ROBOT
# ==========================================
def ejecutar_accion(nombre_accion, repeticiones=1):
    print(f"  [ACCION] {nombre_accion} x{repeticiones}")
    if ROBOT_MODE:
        for _ in range(repeticiones):
            AGC.runActionGroup(nombre_accion)
            time.sleep(0.1)  # pausa entre repeticiones
    else:
        # Modo simulación: solo espera el tiempo equivalente
        if nombre_accion == ACTION_FORWARD:
            time.sleep(TIEMPO_PASO * repeticiones)
        elif nombre_accion in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
            time.sleep(TIEMPO_GIRO_90 * repeticiones)
        else:
            time.sleep(0.5)

def ejecutar_ruta(comandos):
    print("\n[ROBOT] Iniciando recorrido...\n")
    for cmd, cantidad in comandos:
        if cmd == 'forward':
            print(f"  → Avanzar {cantidad} paso(s)")
            ejecutar_accion(ACTION_FORWARD, cantidad)

        elif cmd == 'turn_right':
            grados = cantidad * 90
            print(f"  → Girar derecha {grados}°")
            ejecutar_accion(ACTION_TURN_RIGHT, cantidad)

        elif cmd == 'turn_left':
            grados = cantidad * 90
            print(f"  → Girar izquierda {grados}°")
            ejecutar_accion(ACTION_TURN_LEFT, cantidad)

        time.sleep(0.3)  # pausa de seguridad entre comandos

    print("\n[ROBOT] ¡Meta alcanzada! Deteniendo robot.")
    ejecutar_accion(ACTION_STOP)

# ==========================================
# VISUALIZAR LABERINTO Y RUTA EN CONSOLA
# ==========================================
def visualizar(maze, path):
    path_set = set(path)
    print("\n=== MAPA DEL LABERINTO ===")
    for r, row in enumerate(maze):
        linea = ""
        for c, cell in enumerate(row):
            if (r, c) == INICIO:
                linea += "S"
            elif (r, c) == META:
                linea += "G"
            elif (r, c) in path_set and cell not in ('S','G'):
                linea += "·"
            else:
                linea += cell
        print(linea)
    print(f"\nLongitud de ruta: {len(path)} pasos")
    print("Leyenda: S=Inicio  G=Meta  ·=Ruta  █=Obstáculo\n")

# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 40)
    print("  MechDog — Navegación con A*")
    print("  Modo:", "ROBOT FÍSICO" if ROBOT_MODE else "SIMULACIÓN CONSOLA")
    print("=" * 40)

    # 1. Pararse
    print("\n[INICIO] Levantando robot...")
    ejecutar_accion(ACTION_STAND)
    time.sleep(TIEMPO_STAND)

    # 2. Calcular ruta UNA sola vez
    print("[A*] Calculando ruta...")
    ruta = astar_2d(laberinto_fijo, INICIO, META)

    if not ruta:
        print("[ERROR] No existe ruta desde S hasta G. Revisa el laberinto.")
        return

    print(f"[A*] Ruta encontrada: {len(ruta)} celdas")
    visualizar(laberinto_fijo, ruta)

    # 3. Convertir ruta a comandos de movimiento
    comandos = ruta_a_comandos(ruta)
    print("[PLAN] Secuencia de comandos:")
    for i, (cmd, n) in enumerate(comandos):
        print(f"  {i+1}. {cmd} x{n}")

    # 4. Ejecutar en el robot
    print()
    input("Presiona ENTER para iniciar el recorrido...")
    ejecutar_ruta(comandos)

if __name__ == '__main__':
    main()
