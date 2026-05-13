import numpy as np
import random

class Maze:
    """
    Generador de laberintos usando algoritmo Recursive Backtracking.
    0 = libre, 1 = pared, 2 = inicio, 3 = meta
    """

    def __init__(self, rows=11, cols=11, obstacle_density=0.3, seed=None):
        if rows % 2 == 0:
            rows += 1
        if cols % 2 == 0:
            cols += 1
        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density
        self.seed = seed
        self.grid = None
        self.start = None
        self.goal = None
        self.generate()

    def generate(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.grid = np.ones((self.rows, self.cols), dtype=int)
        self._carve(1, 1)

        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)

        self.grid[self.start] = 2
        self.grid[self.goal] = 3

        self._add_random_obstacles()

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

    def _add_random_obstacles(self):
        free_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
        ]
        n_obstacles = int(len(free_cells) * self.obstacle_density * 0.4)
        chosen = random.sample(free_cells, min(n_obstacles, len(free_cells)))
        for r, c in chosen:
            if (r, c) != self.start and (r, c) != self.goal:
                self.grid[r][c] = 1

    def is_free(self, r, c):
        return (
            0 <= r < self.rows
            and 0 <= c < self.cols
            and self.grid[r][c] != 1
        )

    def get_neighbors(self, r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def reset_marks(self):
        self.grid = np.where(
            (self.grid == 4) | (self.grid == 5),
            0,
            self.grid
        )
        self.grid[self.start] = 2
        self.grid[self.goal] = 3

    def clone_grid(self):
        return self.grid.copy()

    def __repr__(self):
        symbols = {0: ' ', 1: '█', 2: 'S', 3: 'G', 4: '·', 5: '*'}
        rows = []
        for row in self.grid:
            rows.append(''.join(symbols.get(v, '?') for v in row))
        return '\n'.join(rows)
