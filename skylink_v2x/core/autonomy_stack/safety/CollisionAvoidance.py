import numpy as np

class AgentState:
    """ Represents the state of an agent, including its position and velocity. """
    def __init__(self, agent_id, location, speed):
        """
        :param agent_id: int, Unique identifier for the agent
        :param location: tuple(float, float), (x, y) coordinates of the agent
        :param speed: tuple(float, float), (vx, vy) velocity vector of the agent
        """
        self.agent_id = agent_id
        self.location = np.array(location)
        self.speed = np.array(speed)

    def predict_position(self, time_horizon=1.0):
        """
        Predicts the future position of the agent assuming constant velocity.
        :param time_horizon: float, Time step for prediction (in seconds)
        :return: np.array, Predicted (x, y) position after time_horizon
        """
        return self.location + self.speed * time_horizon


class CollisionDetector:
    """ Detects potential collisions between multiple agents. """
    def __init__(self, agents, safe_distance=1.0, prediction_time=1.0):
        """
        :param agents: list[AgentState], List of all agents in the environment
        :param safe_distance: float, Minimum safe distance to avoid collision
        :param prediction_time: float, Time horizon for collision prediction
        """
        self.agents = agents
        self.safe_distance = safe_distance
        self.prediction_time = prediction_time

    def check_collisions(self):
        """
        Checks for potential collisions between all pairs of agents.
        :return: list[tuple(int, int)], List of agent pairs at risk of collision
        """
        collision_pairs = []
        n = len(self.agents)

        for i in range(n):
            for j in range(i + 1, n):
                pos_i = self.agents[i].predict_position(self.prediction_time)
                pos_j = self.agents[j].predict_position(self.prediction_time)
                
                distance = np.linalg.norm(pos_i - pos_j)
                if distance < self.safe_distance:
                    collision_pairs.append((self.agents[i].agent_id, self.agents[j].agent_id))

        return collision_pairs
