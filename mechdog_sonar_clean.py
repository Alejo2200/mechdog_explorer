import heapq
import time
import ujson
import ActionGroupControl  
import Sonar 

laberinto_fijo = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, "S", 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, "G", 1],
    [1, 1, 1, 1, 1, 1, 1]
]

INICIO = (1, 1)  
META   = (5, 5)  

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
                if maze[nb[0]][nb[1]] == 1:
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

def ejecutar_ruta_real(path):
    TIEMPO_PASO_ADELANTE = 0.8  
    TIEMPO_GIRO = 0.6           
    orientacion_actual = 0  

    print("Ruta optima calculada por A*: " + ujson.dumps(path))

    for i in range(1, len(path)):
        r_next, c_next = path[i]
        r_prev, c_prev = path[i-1]

        dr = r_next - r_prev
        dc = c_next - c_prev

        if dc == 1:    orientacion_objetivo = 0  
        elif dr == 1:  orientacion_objetivo = 1  
        elif dc == -1: orientacion_objetivo = 2  
        elif dr == -1: orientacion_objetivo = 3  

        while orientacion_actual != orientacion_objetivo:
            diff = (orientacion_objetivo - orientacion_actual) % 4
            if diff == 1:
                ActionGroupControl.runActionGroup('turn_right')
                time.sleep(TIEMPO_GIRO)
                orientacion_actual = (orientacion_actual + 1) % 4
            elif diff == 3:
                ActionGroupControl.runActionGroup('turn_left')
                time.sleep(TIEMPO_GIRO)
                orientacion_actual = (orientacion_actual - 1) % 4
            elif diff == 2:
                ActionGroupControl.runActionGroup('turn_right')
                time.sleep(TIEMPO_GIRO * 2)
                orientacion_actual = (orientacion_actual + 2) % 4

            ActionGroupControl.runActionGroup('stand')
            time.sleep(0.3)

        distancia = Sonar.getDistance()
        print("Distancia al frente medida por ultrasonido: " + str(distancia) + " cm")

        if distancia < 15.0: 
            print("[ALERTA] Obstaculo imprevisto detectado! Deteniendo motores por seguridad.")
            ActionGroupControl.runActionGroup('stand')
            return False 

        print("Avanzando a celda segura: (" + str(r_next) + "," + str(c_next) + ")")
        ActionGroupControl.runActionGroup('walk_forward')
        time.sleep(TIEMPO_PASO_ADELANTE)
        ActionGroupControl.runActionGroup('stand')
        time.sleep(0.3)
        
    return True

def main():
    print("Inicializando hardware y sensor ultrasonido...")
    Sonar.setRGB(1, 0, 255, 0) 
    ActionGroupControl.runActionGroup('go_up') 
    time.sleep(2.0)

    print("Mapa cargado correctamente via ujson.")
    ruta = astar_2d(laberinto_fijo, INICIO, META)

    if not ruta:
        print("[ERROR] No se pudo calcular la ruta.")
        ActionGroupControl.runActionGroup('stand')
        return

    exito = ejecutar_ruta_real(ruta)
    
    if exito:
        print("Meta alcanzada rapidamente con exito!")
        Sonar.setRGB(1, 0, 0, 255) 
    else:
        print("[FIN] Simulacion pausada por presencia de obstaculos.")
        Sonar.setRGB(1, 255, 0, 0) 
        
    ActionGroupControl.runActionGroup('stand')

if __name__ == '__main__':
    main()
