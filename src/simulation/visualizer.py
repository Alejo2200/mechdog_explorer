import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches

# Colores del mapa
CMAP = mcolors.ListedColormap([
    'white',      # 0 libre
    '#2c2c2c',    # 1 pared
    '#00cc44',    # 2 inicio
    '#ff3333',    # 3 meta
    '#aaddff',    # 4 visitado
    '#ffaa00',    # 5 ruta
])
BOUNDS = [0, 1, 2, 3, 4, 5, 6]
NORM = mcolors.BoundaryNorm(BOUNDS, CMAP.N)

def plot_maze(maze, path=None, visited=None, title="Laberinto", ax=None, show=True):
    grid = maze.clone_grid()

    if visited:
        for r, c in visited:
            if grid[r][c] == 0:
                grid[r][c] = 4

    if path:
        for r, c in path:
            if grid[r][c] not in (2, 3):
                grid[r][c] = 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

    # Leyenda
    patches = [
        mpatches.Patch(color='#00cc44', label='Inicio'),
        mpatches.Patch(color='#ff3333', label='Meta'),
        mpatches.Patch(color='#aaddff', label='Explorado'),
        mpatches.Patch(color='#ffaa00', label='Ruta'),
        mpatches.Patch(color='#2c2c2c', label='Pared'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()

def plot_comparison(maze, astar_result, ql_result, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Comparación A* vs Q-Learning', fontsize=16, fontweight='bold')

    plot_maze(
        maze,
        path=astar_result.get('path'),
        visited=astar_result.get('visited'),
        title=f"A*\nNodos: {astar_result['metrics']['nodes_explored']} | "
              f"Ruta: {astar_result['metrics']['path_length']} | "
              f"Tiempo: {astar_result['metrics']['execution_time']:.4f}s",
        ax=axes[0],
        show=False
    )

    plot_maze(
        maze,
        path=ql_result.get('path'),
        visited=ql_result.get('visited'),
        title=f"Q-Learning\nNodos: {ql_result['metrics']['nodes_explored']} | "
              f"Ruta: {ql_result['metrics']['path_length']} | "
              f"Tiempo: {ql_result['metrics']['execution_time']:.4f}s",
        ax=axes[1],
        show=False
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {save_path}")

    plt.show()

def plot_metrics_bar(results_df, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Métricas Comparativas', fontsize=14, fontweight='bold')

    metrics = ['execution_time', 'path_length', 'nodes_explored']
    titles = ['Tiempo de ejecución (s)', 'Longitud de ruta', 'Nodos explorados']
    colors = ['#4477aa', '#ee7733']

    for ax, metric, title in zip(axes, metrics, titles):
        for i, algo in enumerate(['A*', 'Q-Learning']):
            data = results_df[results_df['algorithm'] == algo][metric]
            ax.bar(algo, data.mean(), color=colors[i], alpha=0.8,
                   yerr=data.std(), capsize=5)
        ax.set_title(title)
        ax.set_ylabel('Promedio')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {save_path}")

    plt.show()

def plot_training_rewards(rewards, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.6, color='steelblue', label='Reward por episodio')

    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg,
                 color='orange', linewidth=2, label=f'Media móvil ({window})')

    plt.title('Curva de entrenamiento Q-Learning')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
