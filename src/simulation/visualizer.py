import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches

# Colores mejorados para el informe
CMAP = mcolors.ListedColormap([
    'white',      # 0 libre
    '#2c3e50',    # 1 pared (Gris oscuro/Azul oscuro)
    '#27ae60',    # 2 inicio (Verde)
    '#e74c3c',    # 3 meta (Rojo)
    '#3498db',    # 4 explorado/mapeado (Azul claro)
    '#f1c40f',    # 5 ruta final (Amarillo/Oro)
])
BOUNDS = [0, 1, 2, 3, 4, 5, 6]
NORM = mcolors.BoundaryNorm(BOUNDS, CMAP.N)

def plot_maze(maze, path=None, visited=None, title="Laberinto", ax=None, show=True):
    grid = maze.clone_grid()

    # Mapeo: Dibujar todas las celdas que el robot logró "ver" o visitar
    if visited:
        for r, c in visited:
            if grid[r][c] == 0:
                grid[r][c] = 4

    # Ruta: Dibujar el camino final encontrado
    if path:
        for r, c in path:
            if grid[r][c] not in (2, 3):
                grid[r][c] = 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    # Leyenda para el informe (Cumple punto 95 del parcial)
    patches = [
        mpatches.Patch(color='#27ae60', label='Inicio'),
        mpatches.Patch(color='#e74c3c', label='Meta'),
        mpatches.Patch(color='#3498db', label='Espacio Mapeado'),
        mpatches.Patch(color='#f1c40f', label='Ruta Optimizada'),
        mpatches.Patch(color='#2c3e50', label='Obstáculo/Pared'),
    ]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=9)

    if show:
        plt.tight_layout()
        plt.show()

def plot_comparison(maze, astar_result, ql_result, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Comparativa de Algoritmos: Exploración y Rutas', fontsize=16, fontweight='bold')

    # Visualización A*
    plot_maze(
        maze,
        path=astar_result.get('path'),
        visited=astar_result.get('visited'),
        title=f"Algoritmo A*\nNodos Mapeados: {astar_result['metrics']['nodes_explored']}\n"
              f"Longitud Ruta: {astar_result['metrics']['path_length']}",
        ax=axes[0],
        show=False
    )

    # Visualización Q-Learning
    plot_maze(
        maze,
        path=ql_result.get('path'),
        visited=ql_result.get('visited'),
        title=f"Algoritmo Q-Learning\nNodos Mapeados: {ql_result['metrics']['nodes_explored']}\n"
              f"Longitud Ruta: {ql_result['metrics']['path_length']}",
        ax=axes[1],
        show=False
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close() # Cerrar para evitar consumo de memoria en WSL

def plot_metrics_bar(results_df, save_path=None):
    # (El resto de funciones como plot_training_rewards se mantienen igual o similares)
    # Solo asegúrate de que usen plt.savefig y plt.close() para WSL
    pass

def plot_training_rewards(rewards, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='#3498db', label='Recompensa/Episodio')
    
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), moving_avg, color='#e67e22', linewidth=2, label='Media Móvil (50)')

    plt.title('Progreso del Aprendizaje (Q-Learning)', fontweight='bold')
    plt.xlabel('Número de Episodio')
    plt.ylabel('Recompensa Acumulada')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()
