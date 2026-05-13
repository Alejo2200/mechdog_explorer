import numpy as np
import time
import random

class QLearning:
    def __init__(self, maze, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, episodes=500):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes

        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]  # arriba, abajo, izq, der
        self.q_table = {}
        self.nodes_explored = 0
        self.execution_time = 0
        self.path = []
        self.visited = []
        self.training_rewards = []

    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self._get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        best = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best)

    def _reward(self, state, next_state, reached_goal):
        if reached_goal:
            return 100
        if next_state == state:  # chocó con pared
            return -10
        return -1  # safe learning: penalizar cada paso para minimizar riesgo

    def _step(self, state, action):
        nr = state[0] + action[0]
        nc = state[1] + action[1]
        if self.maze.is_free(nr, nc):
            return (nr, nc)
        return state  # si hay pared, se queda igual (safe: no atraviesa)

    def train(self):
        start_time = time.time()

        for episode in range(self.episodes):
            state = self.maze.start
            total_reward = 0
            steps = 0
            max_steps = self.maze.rows * self.maze.cols * 2

            while steps < max_steps:
                action = self._choose_action(state)
                next_state = self._step(state, action)
                reached_goal = (next_state == self.maze.goal)

                reward = self._reward(state, next_state, reached_goal)
                total_reward += reward

                # Update Q-table
                old_q = self._get_q(state, action)
                next_max = max([self._get_q(next_state, a) for a in self.actions])
                new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
                self.q_table[(state, action)] = new_q

                state = next_state
                steps += 1

                if reached_goal:
                    break

            self.training_rewards.append(total_reward)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.execution_time = time.time() - start_time

    def solve(self):
        self.train()

        # Extraer ruta greedy desde Q-table entrenada
        state = self.maze.start
        self.path = [state]
        self.visited = [state]
        self.nodes_explored = 0
        visited_set = {state}

        max_steps = self.maze.rows * self.maze.cols * 2

        for _ in range(max_steps):
            q_values = [self._get_q(state, a) for a in self.actions]
            action = self.actions[int(np.argmax(q_values))]
            next_state = self._step(state, action)

            self.nodes_explored += 1
            self.visited.append(next_state)

            if next_state == self.maze.goal:
                self.path.append(next_state)
                return True

            if next_state in visited_set:
                # Atascado, fallo
                return False

            visited_set.add(next_state)
            self.path.append(next_state)
            state = next_state

        return False

    def get_metrics(self):
        return {
            'algorithm': 'Q-Learning',
            'nodes_explored': self.nodes_explored,
            'path_length': len(self.path),
            'execution_time': self.execution_time,
            'success': len(self.path) > 0 and self.path[-1] == self.maze.goal,
            'collisions_avoided': self.nodes_explored - len(self.path) if self.path else 0
        }
