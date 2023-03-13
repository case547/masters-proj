def tyre_pm_reward(ts, alpha=0.5) -> list[float]:
    """Custom reward function returning tyre emissions.
    
    Args:
        ts (TrafficSignal): TrafficSignal object
        alpha (float): scaling factor mapping acceleration to tyre pollution.
    """
    tyre_pm_per_lane = []

    for lane in ts.lanes:
        veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
        acceleration = 0.

        for veh in veh_list:
            veh_lane = ts.sumo.vehicle.getLaneID(veh)
            veh_acc = ts.sumo.vehicle.getAcceleration(veh)
            
            if veh not in ts.env.vehicles:
                ts.env.vehicles[veh] = {veh_lane: veh_acc}
            else:
                ts.env.vehicles[veh][veh_lane] = veh_acc - sum(
                    [ts.env.vehicles[veh][l] for l in ts.env.vehicles[veh].keys() if l != veh_lane]
                )
        
            acceleration += ts.env.vehicles[veh][veh_lane]

    tyre_pm_per_lane.append(alpha * acceleration)

    return -sum(tyre_pm_per_lane)
