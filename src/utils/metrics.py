import pandas as pd
import os

def save_metrics(metrics_list, filepath):
    df = pd.DataFrame(metrics_list)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Métricas guardadas en {filepath}")
    return df

def load_metrics(filepath):
    return pd.read_csv(filepath)

def print_summary(df):
    print("\n========== RESUMEN DE EXPERIMENTOS ==========")
    for algo in df['algorithm'].unique():
        sub = df[df['algorithm'] == algo]
        print(f"\n--- {algo} ---")
        print(f"  Éxito:             {sub['success'].mean()*100:.1f}%")
        print(f"  Tiempo promedio:   {sub['execution_time'].mean():.4f}s")
        print(f"  Ruta promedio:     {sub['path_length'].mean():.1f} pasos")
        print(f"  Nodos explorados:  {sub['nodes_explored'].mean():.1f}")
        print(f"  Colisiones evit.:  {sub['collisions_avoided'].mean():.1f}")
    print("=============================================\n")
