import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.maze import Maze
from src.algorithms.astar import AStar
from src.algorithms.qlearning import QLearning
from src.utils.metrics import save_metrics, print_summary
from src.simulation.visualizer import plot_metrics_bar, plot_comparison
import random

def run_single(maze_params, ql_params=None):
    maze = Maze(**maze_params)

    # A*
    astar = AStar(maze)
    astar_success = astar.solve()
    astar_metrics = astar.get_metrics()
    astar_metrics['maze_rows'] = maze_params['rows']
    astar_metrics['maze_cols'] = maze_params['cols']
    astar_metrics['obstacle_density'] = maze_params.get('obstacle_density', 0.3)

    maze.reset_marks()

    # Q-Learning
    params = ql_params or {}
    ql = QLearning(maze, **params)
    ql_success = ql.solve()
    ql_metrics = ql.get_metrics()
    ql_metrics['maze_rows'] = maze_params['rows']
    ql_metrics['maze_cols'] = maze_params['cols']
    ql_metrics['obstacle_density'] = maze_params.get('obstacle_density', 0.3)

    return {
        'astar': {'metrics': astar_metrics, 'path': astar.path, 'visited': astar.visited},
        'ql':    {'metrics': ql_metrics,    'path': ql.path,    'visited': ql.visited},
        'maze':  maze
    }

def run_all_experiments():
    all_metrics = []
    results_list = []

    experiments = [
        # Variando tamaño
        {'rows': 11, 'cols': 11, 'obstacle_density': 0.1, 'seed': 1},
        {'rows': 11, 'cols': 11, 'obstacle_density': 0.2, 'seed': 2},
        {'rows': 11, 'cols': 11, 'obstacle_density': 0.3, 'seed': 3},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.1, 'seed': 4},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.2, 'seed': 5},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.3, 'seed': 6},
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.1, 'seed': 7},
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.2, 'seed': 8},
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.3, 'seed': 9},
        {'rows': 25, 'cols': 25, 'obstacle_density': 0.1, 'seed': 10},
        # Variando densidad extrema
        {'rows': 11, 'cols': 11, 'obstacle_density': 0.05, 'seed': 11},
        {'rows': 11, 'cols': 11, 'obstacle_density': 0.4,  'seed': 12},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.05, 'seed': 13},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.4,  'seed': 14},
        # Laberintos grandes
        {'rows': 31, 'cols': 31, 'obstacle_density': 0.1, 'seed': 15},
        {'rows': 31, 'cols': 31, 'obstacle_density': 0.2, 'seed': 16},
        # Seeds distintos mismo config
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.2, 'seed': 17},
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.2, 'seed': 18},
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.2, 'seed': 19},
        {'rows': 21, 'cols': 21, 'obstacle_density': 0.2, 'seed': 20},
    ]

    print(f"Ejecutando {len(experiments)} experimentos...\n")

    for i, params in enumerate(experiments):
        print(f"[{i+1:02d}/{len(experiments)}] Tamaño {params['rows']}x{params['cols']} | "
              f"Densidad {params['obstacle_density']} | Seed {params['seed']}")

        result = run_single(params, ql_params={'episodes': 300})
        results_list.append(result)

        all_metrics.append(result['astar']['metrics'])
        all_metrics.append(result['ql']['metrics'])

        a = result['astar']['metrics']
        q = result['ql']['metrics']
        print(f"    A*  → {'✓' if a['success'] else '✗'} | "
              f"Nodos: {a['nodes_explored']:4d} | Ruta: {a['path_length']:3d} | "
              f"Tiempo: {a['execution_time']:.4f}s")
        print(f"    QL  → {'✓' if q['success'] else '✗'} | "
              f"Nodos: {q['nodes_explored']:4d} | Ruta: {q['path_length']:3d} | "
              f"Tiempo: {q['execution_time']:.4f}s")

    # Guardar CSV
    df = save_metrics(all_metrics, 'results/data/experiments.csv')
    print_summary(df)

    # Graficar métricas
    plot_metrics_bar(df, save_path='results/plots/metrics_comparison.png')

    # Graficar primer y último laberinto como ejemplo
    for idx in [0, -1]:
        r = results_list[idx]
        plot_comparison(
            r['maze'], r['astar'], r['ql'],
            save_path=f"results/plots/maze_exp_{idx}.png"
        )

    return df, results_list

if __name__ == '__main__':
    run_all_experiments()
