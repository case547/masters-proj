from sumo_rl import TrafficSignal

def total_abs_accel(ts: TrafficSignal) -> float:
    """Returns the negative of the total absolute acceleration of vehicles in the intersection."""
    reward = 0.
    vehs = ts._get_veh_list()

    for v in vehs:
        accel = ts.sumo.vehicle.getAcceleration(v)
        reward -= abs(accel)

    return reward

def mean_abs_accel(ts: TrafficSignal) -> float:
    """Returns the negative of the mean absolute acceleration of vehicles in the intersection."""
    reward = 0.
    vehs = ts._get_veh_list()

    for v in vehs:
        accel += ts.sumo.vehicle.getAcceleration(v)
        reward -= abs(accel)

    reward /= len(vehs)

    return reward




# Currently, trivial solution exploited where cars are just stopped;
# Therefore need to add extra layer to prevent. Options are:
#   Intersection pressure: num veh leaving minus num veh approaching
#   Diff in waiting time (change in cumulative delay)
#   Normalised average speed
#   Num vehicles queued

# Measure congestion by measuring longest queue - num of vehicles
# waiting time and queue length should be linear anyway
# waiting time maybe more accurate because queue length would have a max