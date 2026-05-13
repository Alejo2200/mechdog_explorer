import sys
import os
import matplotlib
matplotlib.use('Agg')  # Sin necesidad de pantalla, guarda en archivo

from src.simulation.maze import Maze
from src.algorithms.astar import AStar
from src.algorithms.qlearning import QLearning
from src.simulation.visualizer import (
    plot_maze, plot_comparison, plot_training_rewards
)
from src.utils.metrics import print_summary
import pandas as pd

def demo_single(rows=15, cols=15, obstacle_density=0.2, seed=42, ql_episodes=300):
    print(f"\n{'='*50}")
    print(f" DEMO: Laberinto {rows}x{cols} | Densidad {obstacle_density} | Seed {seed}")
    print(f"{'='*50}\n")

    maze = Maze(rows=rows, cols=cols, obstacle_density=obstacle_density, seed=seed)
    print("Laberinto generado:")
    print(maze)
    print()

    # --- A* ---
    print("Ejecutando A*...")
    astar = AStar(maze)
    success = astar.solve()
    m = astar.get_metrics()
    print(f"  Resultado: {'ÉXITO ✓' if success else 'FALLO ✗'}")
    print(f"  Nodos explorados: {m['nodes_explored']}")
    print(f"  Longitud ruta:    {m['path_length']}")
    print(f"  Tiempo:           {m['execution_time']:.4f}s")

    maze.reset_marks()

    # --- Q-Learning ---
    print(f"\nEntrenando Q-Learning ({ql_episodes} episodios)...")
    ql = QLearning(maze, episodes=ql_episodes)
    ql_success = ql.solve()
    qm = ql.get_metrics()
    print(f"  Resultado: {'ÉXITO ✓' if ql_success else 'FALLO ✗'}")
    print(f"  Nodos explorados: {qm['nodes_explored']}")
    print(f"  Longitud ruta:    {qm['path_length']}")
    print(f"  Tiempo:           {qm['execution_time']:.4f}s")

    # Guardar visualizaciones
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    print("\nGenerando imágenes en results/plots/ ...")

    plot_training_rewards(
        ql.training_rewards,
        save_path='results/plots/training_rewards.png'
    )

    plot_comparison(
        maze,
        {'metrics': m, 'path': astar.path, 'visited': astar.visited},
        {'metrics': qm, 'path': ql.path, 'visited': ql.visited},
        save_path='results/plots/demo_comparison.png'
    )

    print("\nImágenes guardadas. Ábrelas desde Windows en:")
    print(r"  \\wsl$\Ubuntu\home\alejo\mechdog_explorer\results\plots")

if __name__ == '__main__':
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    if '--experiments' in sys.argv:
        from experiments.run_experiments import run_all_experiments
        run_all_experiments()
    else:
        demo_single()
