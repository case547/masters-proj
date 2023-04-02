def tyre_pm_reward(ts, alpha=0.5) -> float:
    """Custom reward function returning tyre emissions.
    
    Args:
        ts (TrafficSignal): TrafficSignal object
        alpha (float): scaling factor mapping acceleration to tyre pollution.
    """
    tyre_pm_per_lane = []

    for ts_lane in ts.lanes:
        # Get IDs of vehicles on this lane in last simulation step 
        veh_list = ts.sumo.lane.getLastStepVehicleIDs(ts_lane)
        
        lane_tyre_pm = 0.

        for veh in veh_list:
            veh_acc = ts.sumo.vehicle.getAcceleration(veh)
            lane_tyre_pm += alpha * abs(veh_acc)

        tyre_pm_per_lane.append(lane_tyre_pm)

    return -sum(tyre_pm_per_lane)
