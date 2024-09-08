import random
import numpy as np
import copy

# replay_memory.py
class ReplayMemory:
    def __init__(self, capacity, seed, task_count):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.task_rewards = {i: [] for i in range(task_count)}  # 작업별 보상 버퍼 추가
        self.done_task = []
        self.task_count = task_count

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def push_task(self, episode_reward, task_id):
        self.task_rewards[task_id].append(episode_reward)  # 작업별 보상 버퍼에 보상 추가
        if len(self.task_rewards[task_id]) > 20:  # 보상 버퍼 크기를 100으로 제한
            self.task_rewards[task_id].pop(0)

    def sample_task(self):
        t_rewards = {i: [] for i in range(self.task_count)}
        for task_id in range(self.task_count):
            if len(self.task_rewards[task_id]) > 0:
                t_rewards[task_id] = copy.deepcopy(self.task_rewards[task_id])

        for done_index in self.done_task:
            del t_rewards[int(done_index)]

        task_id = random.choice(list(t_rewards.keys()))
        if len(self.task_rewards[task_id]) > 0:
            return task_id, random.choice(self.task_rewards[task_id])
        else:
            return task_id, 1  # 보상이 없을 경우 기본 보상

    def __len__(self):
        return len(self.buffer)
