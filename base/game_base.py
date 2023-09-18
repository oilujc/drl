from abc import ABC, abstractmethod

class GameBase(ABC):

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def update(self, action):
        pass

    @abstractmethod
    def get_state(self):
        pass
    
    # @abstractmethod
    # def get_reward(self):
    #     pass
    
    # @abstractmethod
    # def get_done(self):
    #     pass
    
    # @abstractmethod
    # def get_info(self):
    #     pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def render(self):
        pass
