from sumo_rl import TrafficSignal


def get_total_waiting_time(ts: TrafficSignal) -> float:
    """Return the waiting time for all vehicles in the intersections.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    return sum(ts.sumo.lane.getWaitingTime(lane) for lane in ts.lanes)


def get_tyre_pm(ts: TrafficSignal) -> float:
    """Return tyre PM emission based on absolute acceleration.
    
    Tyre PM emission and vehicle absolute acceleration are assumed to have a linear relationship.

    Keyword arguments
        ts: the TrafficSignal object
    """
    tyre_pm = 0.

    for lane in ts.lanes:
        veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
        
        for veh in veh_list:
            accel = ts.sumo.vehicle.getAcceleration(veh)
            # Assume acceleration has been constant for the past delta_time seconds
            tyre_pm += abs(accel) * ts.delta_time

    return tyre_pm
