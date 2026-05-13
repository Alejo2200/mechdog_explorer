import heapq
import time

class AStar:
    def __init__(self, maze):
        self.maze = maze
        self.nodes_explored = 0
        self.execution_time = 0
        self.path = []
        self.visited = []

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        self.nodes_explored = 0
        self.path = []
        self.visited = []

        start = self.maze.start
        goal = self.maze.goal

        start_time = time.time()

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            self.nodes_explored += 1
            self.visited.append(current)

            if current == goal:
                self.execution_time = time.time() - start_time
                self.path = self._reconstruct(came_from, current)
                return True

            for neighbor in self.maze.get_neighbors(*current):
                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        self.execution_time = time.time() - start_time
        return False

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_metrics(self):
        return {
            'algorithm': 'A*',
            'nodes_explored': self.nodes_explored,
            'path_length': len(self.path),
            'execution_time': self.execution_time,
            'success': len(self.path) > 0,
            'collisions_avoided': self.nodes_explored - len(self.path) if self.path else 0
        }
