from gymnasium import spaces
import numpy as np
from sumo_rl.environment.observations import DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from supersuit.utils.action_transforms.homogenize_ops import pad_to


grid2x2_neighbours = {
    '1': ['2', '5'],
    '2': ['1', '6'],
    '5': ['1', '6'],
    '6': ['2', '5'],
}

grid4x4_neighbours = {
    'A0': ['A1','B0'],
    'A1': ['A0','A2','B1'],
    'A2': ['A1','A3','B2'],
    'A3': ['A2','B3'],

    'B0': ['A0','B1','C0'],
    'B1': ['A1','B0','B2','C0'],
    'B2': ['A2','B1','B3','C2'],
    'B3': ['A3','B2','C3'],

    'C0': ['B0','C1','D0'],
    'C1': ['B1','C0','C2','D0'],
    'C2': ['B2','C1','C3','D2'],
    'C3': ['B3','C2','D3'],

    'D0': ['C0','D1'],
    'D1': ['C1','D0','D2'],
    'D2': ['D1','C2','D3'],
    'D3': ['D2','C3'],
}

def max_neighbours(neighbours: dict):
    return max([len(v) for v in neighbours.values()])


class SharedObservationFunction(DefaultObservationFunction):
    """Class to share observations between neighbouring traffic signals in multi-agent networks."""
    
    def __init__(self, ts: TrafficSignal, neighbour_dict: dict):
        """Initialise observation function."""
        super().__init__(ts)
        self.neighbour_dict = neighbour_dict
        self.num_neighbours = len(neighbour_dict[self.ts.id])

    def __call__(self) -> np.ndarray:
        obs = self.ts._observation_fn_default()

        neighbour: TrafficSignal  # type hint for VS Code

        # Below: checks for self.neighbours before self.traffic_signals because if
        # self.traffic_signals exists, then self.neighbours must have been created

        if hasattr(self, "neighbours"):
            for neighbour in self.neighbours:
                obs = np.hstack((obs, neighbour._observation_fn_default()))

            return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        
        if hasattr(self.ts.env, "traffic_signals"):
            self.neighbours = [self.ts.env.traffic_signals[n_id] for n_id in self.neighbour_dict[self.ts.id]]

        return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        self.space_dim = (self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)) \
                          * (1 + max_neighbours(self.neighbour_dict))

        return spaces.Box(
            low=np.zeros(self.space_dim, dtype=np.float32),
            high=np.ones(self.space_dim, dtype=np.float32),
        )
    

class Grid2x2ObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts, grid2x2_neighbours)


class Grid4x4ObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts, grid4x4_neighbours)