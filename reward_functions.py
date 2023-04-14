from sumo_rl import TrafficSignal

def tyre_pm_reward(ts: TrafficSignal, alpha=1.0) -> float:
    """Return the reward based on tyre PM emission.
    
    Tyre PM emission is assumed to be linearly related to the vehicles'
    absolute acceleration, and the penalty is calculated as the negative
    of the total absolute acceleration of vehicles in the intersection.
    
    Keyword arguments
        ts: the TrafficSignal object
        alpha: coefficient mapping acceleration to tyre PM emission (default 1.0)
    """
    tyre_pm = 0.
    vehs = ts._get_veh_list()

    for v in vehs:
        accel = ts.sumo.vehicle.getAcceleration(v)
        tyre_pm -= alpha * abs(accel)

    return tyre_pm

def tyre_pm_and_delay(ts: TrafficSignal, alpha=1.0) -> float:
    """Return the reward summing tyre PM and change in total delay.
    
    Keyword arguments
        ts: the TrafficSignal object
        alpha: coefficient mapping acceleration to tyre PM emission (default 1.0)
    """
    accel_reward = tyre_pm_reward(ts, alpha)

    accumulated_wait = -sum(ts.get_accumulated_waiting_time_per_lane()) / 100
    wait_time_reward = ts.last_measure - accumulated_wait
    ts.last_measure = accumulated_wait

    return accel_reward + wait_time_reward




# Currently, trivial solution exploited where cars are just stopped;
# Therefore need to add extra layer to prevent. Options are:
#   Intersection pressure: num veh leaving minus num veh approaching
#   Diff in waiting time (change in cumulative delay)
#   Normalised average speed
#   Num vehicles queued

# Measure congestion by measuring longest queue - num of vehicles
# waiting time and queue length should be linear anyway
# waiting time maybe more accurate because queue length would have a max