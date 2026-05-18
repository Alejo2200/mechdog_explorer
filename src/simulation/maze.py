import numpy as np
import random
from collections import deque

class Maze:
    """
    Generador de laberintos con verificación de ruta para garantizar 
    que la meta sea alcanzable (Requisito de funcionalidad).
    """

    def __init__(self, rows=11, cols=11, obstacle_density=0.3, seed=None):
        if rows % 2 == 0: rows += 1
        if cols % 2 == 0: cols += 1
        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density
        self.seed = seed
        self.grid = None
        self.start = (1, 1)
        self.goal = (rows - 2, cols - 2)
        self.generate()

    def generate(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 1. Crear estructura básica con paredes
        self.grid = np.ones((self.rows, self.cols), dtype=int)
        self._carve(1, 1)

        # 2. Asegurar inicio y meta
        self.grid[self.start] = 2
        self.grid[self.goal] = 3

        # 3. Añadir obstáculos extra con validación de ruta
        self._add_safe_obstacles()

    def _carve(self, r, c):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        self.grid[r][c] = 0
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < self.rows - 1 and 0 < nc < self.cols - 1:
                if self.grid[nr][nc] == 1:
                    self.grid[r + dr // 2][c + dc // 2] = 0
                    self._carve(nr, nc)

    def _has_path(self):
        """Verifica con BFS si existe camino entre inicio y meta."""
        queue = deque([self.start])
        visited = {self.start}
        while queue:
            curr = queue.popleft()
            if curr == self.goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                    self.grid[nr][nc] != 1 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _add_safe_obstacles(self):
        """Añade obstáculos solo si no bloquean la salida."""
        free_cells = [
            (r, c) for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
            if self.grid[r][c] == 0 and (r, c) != self.start and (r, c) != self.goal
        ]
        
        # Intentamos añadir el 40% de la densidad solicitada como paredes extra
        n_to_add = int(len(free_cells) * self.obstacle_density * 0.4)
        random.shuffle(free_cells)
        
        added = 0
        for r, c in free_cells:
            if added >= n_to_add:
                break
            # Guardar estado original
            self.grid[r][c] = 1
            if self._has_path():
                added += 1
            else:
                # Si bloquea el camino, revertir
                self.grid[r][c] = 0

    def is_free(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 1

    def get_neighbors(self, r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def reset_marks(self):
        self.grid = np.where((self.grid == 4) | (self.grid == 5), 0, self.grid)
        self.grid[self.start] = 2
        self.grid[self.goal] = 3

    def clone_grid(self):
        return self.grid.copy()

    def __repr__(self):
        symbols = {0: ' ', 1: '█', 2: 'S', 3: 'G', 4: '·', 5: '*'}
        return '\n'.join(''.join(symbols.get(v, '?') for v in row) for row in self.grid)
