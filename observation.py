from gymnasium import spaces
import numpy as np
from sumo_rl.environment.observations import DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from supersuit.utils.action_transforms.homogenize_ops import pad_to


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
max_neighbours = max([len(v) for v in grid4x4_neighbours.values()])

class Grid4x4ObservationFunction(DefaultObservationFunction):
    """Observation function for traffic signals in the grid4x4 network."""

    def __init__(self, ts: TrafficSignal):
        """Initialise observation function."""
        super().__init__(ts)
        self.num_neighbours = len(grid4x4_neighbours[self.ts.id])

    def __call__(self) -> np.ndarray:
        observation = list(self.ts._observation_fn_default())

        neighbour: TrafficSignal
        if hasattr(self, "neighbours"):
            for neighbour in self.neighbours:
                observation += list(neighbour._observation_fn_default())

            observation = np.array(observation, dtype=np.float32)
            return pad_to(observation, np.zeros(self.space_dim).shape, 0)
        
        # Skipped if self.neighbours already exists to save time
        if hasattr(self.ts.env, "traffic_signals"):
            self.neighbours = [self.ts.env.traffic_signals[n_id] for n_id in grid4x4_neighbours[self.ts.id]]

            for neighbour in self.neighbours:
                observation += list(neighbour._observation_fn_default())

        observation = np.array(observation, dtype=np.float32)
        return pad_to(observation, np.zeros(self.space_dim).shape, 0)
        
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        self.space_dim = (self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)) * (1 + max_neighbours)

        return spaces.Box(
            low=np.zeros(self.space_dim, dtype=np.float32),
            high=np.ones(self.space_dim, dtype=np.float32),
        )