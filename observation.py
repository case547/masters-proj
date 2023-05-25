import numpy as np
from sumo_rl.environment.observations import DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal


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

class Grid4x4ObservationFunction(DefaultObservationFunction):
    """Observation function for traffic signals in the grid4x4 network."""

    def __init__(self, ts: TrafficSignal):
        """Initialise observation function."""
        self.ts = ts

        if hasattr(self.ts.env, "traffic_signals"):
            self.neighbours = [self.ts.env.traffic_signals[n_id] for n_id in grid4x4_neighbours[self.ts.id]]

    def default_observation(self) -> np.ndarray:
        """Return the default observation."""
        return super().__call__()        

    def __call__(self) -> np.ndarray:
        observation = list(self.default_observation())

        if hasattr(self, "neighbours"):
            for neighbour in self.neighbours:
                phase_id = [1 if neighbour.green_phase == i else 0 for i in range(neighbour.num_green_phases)]  # one-hot encoding
                min_green = [0 if neighbour.time_since_last_phase_change < neighbour.min_green + neighbour.yellow_time else 1]
                density = neighbour.get_lanes_density()
                queue = neighbour.get_lanes_queue()

                observation += phase_id + min_green + density + queue

        return np.array(observation, dtype=np.float32)