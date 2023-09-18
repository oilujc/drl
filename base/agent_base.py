from abc import ABC, abstractmethod

class AgentBase(ABC):

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def train(self, iter):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass