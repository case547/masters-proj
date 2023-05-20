from sumo_rl import TrafficSignal

from helper_functions import get_total_waiting_time, get_tyre_pm


def tyre_pm_reward(ts: TrafficSignal) -> float:
    """Return the reward as the amount of tyre PM emitted.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    return -get_tyre_pm(ts)


def diff_accum_wait_time(ts: TrafficSignal) -> float:
    """Return the reward as the change in total cumulative delays.

    The total cumulative delay at time `t` is the sum of the accumulated wait time
    of all vehicles present, from `t = 0` to current time step `t` in the system.

    See https://arxiv.org/abs/1704.08883
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    # ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    # congestion_reward = ts.last_measure - ts_wait
    # ts.last_measure = ts_wait

    # return tyre_pm(ts) + congestion_reward
    return NotImplementedError


def delta_wait_time_reward(ts: TrafficSignal) -> float:
    """Return the reward as change in total waiting time.

    Waiting time is the consecutive time (in seconds) where a vehicle has been standing, exlcuding
    voluntary stopping. See https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getWaitingTime
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = get_total_waiting_time(ts)
    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    return reward


def combined_reward(ts: TrafficSignal, alpha=0.9) -> float:
    """Return the reward summing tyre PM and change in total waiting time.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    return tyre_pm_reward(ts) + alpha*delta_wait_time_reward(ts)

# Currently, trivial solution exploited where cars are just stopped;
# Therefore need to add extra layer to prevent. Options are:
#   Intersection pressure: num veh leaving minus num veh approaching
#   Diff in waiting time (change in cumulative delay)
#   Normalised average speed
#   Num vehicles queued

# Measure congestion by measuring longest queue - num of vehicles
# waiting time and queue length should be linear anyway
# waiting time maybe more accurate because queue length would have a max