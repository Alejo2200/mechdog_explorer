import numpy as np
import time
import random

class QLearning:
    def __init__(self, maze, alpha=0.2, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, episodes=500):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes

        self.actions = [0, 1, 2, 3]
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.q_table = {}
        self.path = []
        self.visited = []
        self.training_rewards = []
        self.execution_time = 0
        self._trained = False

    def _get_reward(self, state, next_state, hit_wall):
        if hit_wall:
            return -15
        if next_state == self.maze.goal:
            return 500
        dist_before = abs(state[0] - self.maze.goal[0]) + abs(state[1] - self.maze.goal[1])
        dist_after  = abs(next_state[0] - self.maze.goal[0]) + abs(next_state[1] - self.maze.goal[1])
        return -1 + (dist_before - dist_after) * 2

    def train(self):
        start_t = time.time()
        max_steps = self.maze.rows * self.maze.cols * 2

        for ep in range(self.episodes):
            state = self.maze.start
            episode_reward = 0

            for _ in range(max_steps):
                if random.random() < self.epsilon:
                    action = random.choice(self.actions)
                else:
                    q_vals = [self.q_table.get((state, a), 0.0) for a in self.actions]
                    action = int(np.argmax(q_vals))

                dr, dc = self.moves[action]
                candidate = (state[0] + dr, state[1] + dc)
                hit_wall = not self.maze.is_free(candidate[0], candidate[1])
                next_state = state if hit_wall else candidate

                reward = self._get_reward(state, next_state, hit_wall)
                episode_reward += reward

                old_q    = self.q_table.get((state, action), 0.0)
                next_max = max([self.q_table.get((next_state, a), 0.0) for a in self.actions])
                self.q_table[(state, action)] = old_q + self.alpha * (
                    reward + self.gamma * next_max - old_q
                )

                state = next_state
                if state == self.maze.goal:
                    break

            self.training_rewards.append(episode_reward)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.execution_time = time.time() - start_t
        self._trained = True

    def solve(self):
        if not self._trained:
            self.train()

        state = self.maze.start
        self.path = [state]
        self.visited = [state]
        visited_set = {state}
        max_steps = self.maze.rows * self.maze.cols * 3

        for _ in range(max_steps):
            q_vals = [self.q_table.get((state, a), 0.0) for a in self.actions]
            action = int(np.argmax(q_vals))
            dr, dc = self.moves[action]
            next_state = (state[0] + dr, state[1] + dc)

            if not self.maze.is_free(next_state[0], next_state[1]):
                break
            if next_state in visited_set:
                break

            visited_set.add(next_state)
            self.visited.append(next_state)
            self.path.append(next_state)
            state = next_state

            if state == self.maze.goal:
                return True

        return False

    def get_metrics(self):
        success = len(self.path) > 1 and self.path[-1] == self.maze.goal
        return {
            'algorithm': 'Q-Learning',
            'nodes_explored': len(self.visited),
            'path_length': len(self.path) if success else 0,
            'execution_time': self.execution_time,
            'success': success,
            'collisions_avoided': len(self.visited)
        }
