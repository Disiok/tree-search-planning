import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDM_DICT
from highway_env.vehicle.objects import Obstacle, Landmark


class CrossMergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    COLLISION_REWARD: float = -3.
    RIGHT_LANE_REWARD: float = 0.
    HIGH_SPEED_REWARD: float = 0
    MERGING_SPEED_REWARD: float = -0.0
    LANE_CHANGE_REWARD: float = 0.0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        
        cross_merge_config = {
            'road_length': 300,
            'num_lanes': 2,
            'actor_density': 0.5,
            'actor_speed_mean': 20,
            'actor_speed_std': 2,
            'actor_spawn_sep': 15,
            'veh_type': 'Polite',
            'goal_reward': 1,
            'goal_radius': 3,
        }
        config.update(cross_merge_config)

        return config

    def step(self, *args, **kwargs):
        self.prev_ego_pos = np.array(self.vehicle.position)
        #n_crashed = self._other_crashed()
        #if n_crashed > 0:
        #    print(f"{n_crashed} othere vehicles crashed")
        
        obs, reward, terminal, info = super().step(*args, **kwargs)
        info["num_goals_reached"] = len(self.reached_goals)
        info["speed"] = self.vehicle.speed
        info["reward"] = reward

        return obs, reward, terminal, info
    
    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

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

        # Ego goal
        ego_position = self.vehicle.position

        for idx, goal in enumerate(self.goals):
            if goal in self.reached_goals:
                continue
            goal_position = goal.position
            #dist = np.linalg.norm(ego_position - goal_position)
            dist = point2line_dist(goal_position, self.prev_ego_pos, ego_position)    
        
            if dist < self.config['goal_radius']:
                reward += self.config['goal_reward']
                self.reached_goals.append(goal)
                if len(self.reached_goals) == 2:
                    print("both goals reached!!")
                #self.goals.pop(idx)
                break
        
        return action_reward[action] + reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > self.road_len

    def _other_crashed(self) -> int:
        return sum([v.crashed for v in self.road.vehicles])

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.prev_ego_pos = None
        self.reached_goals = []

    def _make_straight_roads(self, layout, xl, yl, y0=0, x0=0, width=None):
        """
        Make parallel straight roads
        layout: [0, 1] x N 
        Return road nodes
        """
        if width is None:
            width = StraightLane.DEFAULT_WIDTH

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

            lane = StraightLane([x0, y], [x0 + xl, y + yl], line_types=line_type, width=width)    
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
        lengths = [0.45, 0.1, 0.1, 0.1, 0.35]
        lengths = [l * self.config['road_length'] for l in lengths]

        self.road_len = sum(lengths)
        x = 0
        idx = 0
        ydiff = 4
        n_lanes = self.config['num_lanes']

        goal_positions = []
    
        straights = self._make_straight_roads([1, 1] * n_lanes, lengths[idx], 0, x0=0, width=StraightLane.DEFAULT_WIDTH)
        for s in straights:
            net.add_lane("a", "b", s)
        # goal_positions.append(np.random.choice(straights[-1].position(lengths[idx] * 0.8, 0))
        
        # TODO clean this
        x += lengths[idx]
        idx += 1
        diag0 = self._make_straight_roads([1] * n_lanes, lengths[idx], -ydiff, x0=x)
        diag1 = self._make_straight_roads([1] * n_lanes, lengths[idx], ydiff, x0=x, y0=StraightLane.DEFAULT_WIDTH * n_lanes)
        
        for d in diag0:
            net.add_lane("b", "d0", d)
        for d in diag1:
            net.add_lane("b", "d1", d)
        #goal_positions.append(np.random.choice(diag0 + diag1).position(lengths[idx] * 0.8, 0))

        x += lengths[idx]
        idx += 1
        straight0 = self._make_straight_roads([1] * n_lanes, lengths[idx], 0, x0=x, y0=-ydiff)
        straight1 = self._make_straight_roads([1] * n_lanes, lengths[idx], 0, x0=x, y0=ydiff+StraightLane.DEFAULT_WIDTH * n_lanes)
        for s in straight0:
            net.add_lane("d0", "e0", s)
        for s in straight1:
            net.add_lane("d1", "e1", s)
        #goal_positions.append(np.random.choice(straight1).position(lengths[idx] * 0.5, 0))
        goal_positions.append(straight1[-1].position(lengths[idx] * 0.5, 0))
        
        x += lengths[idx]
        idx += 1
        diag0 = self._make_straight_roads([1] * n_lanes, lengths[idx], ydiff, x0=x, y0=-ydiff)
        diag1 = self._make_straight_roads([1] * n_lanes, lengths[idx], -ydiff, x0=x, y0=ydiff+StraightLane.DEFAULT_WIDTH * n_lanes)
        for d in diag0:
            net.add_lane("e0", "f", d)
        for d in diag1:
            net.add_lane("e1", "f", d)
        #goal_positions.append(np.random.choice(diag0 + diag1).position(lengths[idx] * 0.8, 0))
        
        x += lengths[idx]
        idx += 1
        straights = self._make_straight_roads([1, 1] * n_lanes, lengths[idx], 0, x0=x, width=StraightLane.DEFAULT_WIDTH)
        for s in straights:
            net.add_lane("f", "g", s)
        goal_positions.append(straights[0].position(lengths[idx] * 0.9, 0))

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
        sep = self.config['actor_spawn_sep']
        
        ego_spawn = road.network.get_lane(('a', 'b', 0)).position(sep, 0)
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     ego_spawn)
        road.vehicles.append(ego_vehicle)

        spawns = []

        for lane in road.network.lanes_list():
            for xi in range(int(lane.length / sep)):
                pos = lane.position(xi * sep, 0)
                if pos[0] * 1000 + pos[1] == ego_spawn[0] * 1000 + ego_spawn[1]:
                    continue
                if pos[0] > 0.35 * self.config['road_length']:
                    continue
                spawns.append(pos)

        nv = len(spawns) * self.config['actor_density']
        nv = int(nv)
        np.random.shuffle(spawns)
        spawns = spawns[:nv]

        veh_type = IDM_DICT.get(self.config['veh_type'])

        for spawn in spawns:
            veh = veh_type(road, spawn, heading=0, speed=np.random.normal(self.config['actor_speed_mean'], self.config['actor_speed_std']))
            road.vehicles.append(veh)
        
        self.vehicle = ego_vehicle
        
        return 


def point2line_dist(p, v1, v2):
    v1p = p - v1
    v2p = p - v2

    vp, fp = (v1p, v2p) if np.linalg.norm(v1p) < np.linalg.norm(v2p) else (v2p, v1p)
    vv = vp - fp
    pvv = np.dot(vv, vp)

    vv_norm = np.linalg.norm(vv)
    vp_proj = np.dot(vp, vv) / vv_norm
    

    if pvv > 0:
        dist = np.sqrt(-(vp_proj ** 2) + np.linalg.norm(vp) ** 2)
    else:
        dist = np.linalg.norm(vp)

    #print(p, v1, v2)
    #print('dist', dist)

    return  dist


register(
    id='cross-merge-v0',
    entry_point='highway_env.envs:CrossMergeEnv',
)
