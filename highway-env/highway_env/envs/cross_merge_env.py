import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle, Landmark


class CrossMergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    COLLISION_REWARD: float = -3
    RIGHT_LANE_REWARD: float = 0.
    HIGH_SPEED_REWARD: float = 0.2
    MERGING_SPEED_REWARD: float = -0.0
    LANE_CHANGE_REWARD: float = -0.05

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        
        cross_merge_config = {
            'road_length': 300,
            'num_lanes': 2,
            'actor_density': 2,
            'actor_speed_mean': 30,
            'actor_speed_std': 7,
            'goal_reward': 2,
            'goal_radius': 3,
        }

        config.update(cross_merge_config)

        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
                 + self.HIGH_SPEED_REWARD * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_SPEED_REWARD * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        # Ego goal
        ego_position = self.vehicle.position
        dists = []

        for goal in self.goals:
            goal_position = goal.position
            dist = np.linalg.norm(ego_position - goal_position)
            dists.append(dist)

        dist = min(dists)

        if dist < self.config['goal_radius']:
            reward += self.config['goal_reward']

        return utils.lmap(action_reward[action] + reward,
                          [self.COLLISION_REWARD + self.MERGING_SPEED_REWARD,
                            self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > self.road_len

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_straight_roads(self, layout, xl, yl, y0=0, x0=0):
        """
        Make parallel straight roads
        layout: [0, 1] x N 
        Return road nodes
        """

        y = y0
        lanes = []

        for i, l in enumerate(layout):
            if l == 0:
                y += StraightLane.DEFAULT_WIDTH
                continue
            
            line_type = []
            if i == 0 or layout[i-1] == 0:
                line_type.append(LineType.CONTINUOUS_LINE)
            else:
                line_type.append(LineType.NONE)

            if i == len(layout) - 1 or layout[i+1] == 0:
                line_type.append(LineType.CONTINUOUS_LINE)
            else:
                line_type.append(LineType.STRIPED)

            lane = StraightLane([x0, y], [x0 + xl, y + yl], line_types=line_type)    
            lanes.append(lane)

            y += StraightLane.DEFAULT_WIDTH
            
        return lanes

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        lengths = [0.3, 0.15, 0.15, 0.15, 0.25]
        lengths = [l * self.config['road_length'] for l in lengths]

        self.road_len = sum(lengths)
        x = 0
        idx = 0
        n_lanes=2

        goal_positions = []
    
        straights = self._make_straight_roads([1,1] * n_lanes, lengths[idx], 0, x0=0)
        for s in straights:
            net.add_lane("a", "b", s)
        goal_positions.append(np.random.choice(straights).position(lengths[idx] * 0.8, 0))
        
        # TODO clean this
        x += lengths[idx]
        idx += 1
        diag0 = self._make_straight_roads([1] * n_lanes, lengths[idx], -4, x0=x)
        diag1 = self._make_straight_roads([1] * n_lanes, lengths[idx], 4, x0=x, y0=StraightLane.DEFAULT_WIDTH * n_lanes)
        
        for d in diag0:
            net.add_lane("b", "d0", d)
        for d in diag1:
            net.add_lane("b", "d1", d)
        goal_positions.append(np.random.choice(diag0 + diag1).position(lengths[idx] * 0.8, 0))

        x += lengths[idx]
        idx += 1
        straight0 = self._make_straight_roads([1] * n_lanes, lengths[idx], 0, x0=x, y0=-4)
        straight1 = self._make_straight_roads([1] * n_lanes, lengths[idx], 0, x0=x, y0=4+StraightLane.DEFAULT_WIDTH * n_lanes)
        for s in straight0:
            net.add_lane("d0", "e0", s)
        for s in straight1:
            net.add_lane("d1", "e1", s)
        goal_positions.append(np.random.choice(straight0 + straight1).position(lengths[idx] * 0.8, 0))
        
        x += lengths[idx]
        idx += 1
        diag0 = self._make_straight_roads([1] * n_lanes, lengths[idx], 4, x0=x, y0=-4)
        diag1 = self._make_straight_roads([1] * n_lanes, lengths[idx], -4, x0=x, y0=4+StraightLane.DEFAULT_WIDTH * n_lanes)
        for d in diag0:
            net.add_lane("e0", "f0", d)
        for d in diag1:
            net.add_lane("e1", "f1", d)
        goal_positions.append(np.random.choice(diag0 + diag1).position(lengths[idx] * 0.8, 0))
        
        x += lengths[idx]
        idx += 1
        straights = self._make_straight_roads([1,1] * n_lanes, lengths[idx], 0, x0=x)
        for s in straights[:n_lanes]:
            net.add_lane("f0", "g", s)
        for s in straights[n_lanes:]:
            net.add_lane("f1", "g", s)
        goal_positions.append(np.random.choice(straights).position(lengths[idx] * 0.8, 0))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        
        #set goals
        self.goals = []

        for pos in goal_positions:
            goal = Landmark(road, pos)
            self.goals.append(goal)
            road.objects.append(goal)

        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)
        
        nv = int(self.config['actor_density'] * self.config['num_lanes'] * 2) 
        n_per_lane = 3

        # hardcoded spawning
        ns = n_per_lane * 2 * self.config['num_lanes']
        locs = list(range(ns))
        locs.pop(2 * n_per_lane - 1)

        for idx in range(nv):
            loc = locs.pop(np.random.randint(len(locs)))
            lane_idx = loc // n_per_lane
            lane_pos = 30 * (loc % n_per_lane)

            veh = IDMVehicle(road, road.network.get_lane(("a", "b", lane_idx)).position(lane_pos, 0), heading=0, speed=np.random.normal(self.config['actor_speed_mean'], self.config['actor_speed_std']))
            road.vehicles.append(veh)
        
        self.vehicle = ego_vehicle
        
        return 


register(
    id='cross-merge-v0',
    entry_point='highway_env.envs:CrossMergeEnv',
)
