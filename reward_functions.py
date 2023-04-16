from helper_functions import get_tyre_pm
from sumo_rl import TrafficSignal


def tyre_pm_reward(ts: TrafficSignal, alpha=1.0) -> float:
    """Return the reward as the amount of tyre PM emitted.
    
    Keyword arguments
        ts: the TrafficSignal object
        alpha: coefficient mapping acceleration to tyre PM emission (default 1.0)
    """
    return -get_tyre_pm(ts, alpha)


def tyre_pm_and_diff_wait_time(ts: TrafficSignal, alpha=1.0) -> float:
    """Return the reward summing tyre PM and change in total delay.
    
    Keyword arguments
        ts: the TrafficSignal object
        alpha: coefficient mapping acceleration to tyre PM emission (default 1.0)
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    diff_wait_time_reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    return tyre_pm_reward(ts, alpha) + diff_wait_time_reward



# Currently, trivial solution exploited where cars are just stopped;
# Therefore need to add extra layer to prevent. Options are:
#   Intersection pressure: num veh leaving minus num veh approaching
#   Diff in waiting time (change in cumulative delay)
#   Normalised average speed
#   Num vehicles queued

# Measure congestion by measuring longest queue - num of vehicles
# waiting time and queue length should be linear anyway
# waiting time maybe more accurate because queue length would have a max