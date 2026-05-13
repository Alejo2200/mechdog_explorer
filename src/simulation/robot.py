class Robot:
    """
    Modelo del robot MechDog en simulación.
    Simula sensor ultrasónico con rango máximo.
    """

    SENSOR_RANGE = 3  # celdas que puede "ver" el sensor

    def __init__(self, maze):
        self.maze = maze
        self.position = maze.start
        self.path_taken = []
        self.collisions_avoided = 0

    def reset(self):
        self.position = self.maze.start
        self.path_taken = []
        self.collisions_avoided = 0

    def sense(self):
        """Simula el sensor ultrasónico: devuelve obstáculos en rango."""
        r, c = self.position
        obstacles = []
        for dr in range(-self.SENSOR_RANGE, self.SENSOR_RANGE + 1):
            for dc in range(-self.SENSOR_RANGE, self.SENSOR_RANGE + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols:
                    if self.maze.grid[nr][nc] == 1:
                        dist = abs(dr) + abs(dc)
                        obstacles.append(((nr, nc), dist))
        return obstacles

    def move(self, new_position):
        """
        Safe learning: solo se mueve si la celda es libre.
        Si hay obstáculo, cuenta como colisión evitada.
        """
        if self.maze.is_free(*new_position):
            self.path_taken.append(self.position)
            self.position = new_position
            return True
        else:
            self.collisions_avoided += 1
            return False

    def at_goal(self):
        return self.position == self.maze.goal
