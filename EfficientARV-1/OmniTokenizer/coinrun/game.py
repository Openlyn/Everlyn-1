# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import json


class Game:
    def __init__(self, **kwargs):
        self.game_id = -1
        self.level_seed = 0
        self.rl_agent_seed = 0
        self.zoom = 5.5
        self.bgzoom = 0.4    # NOTE: hard-coded
        self.world_theme_n = -1
        self.agent_theme_n = -1

        self.background_themes = []
        self.ground_themes = []
        self.agent_themes = []
        self.monster_names = {}
        self.flattened_monster_names = []

        # TODO: save and load these from the game engine
        self.video_res = 1024
        self.maze_w = 64
        self.maze_h = 13  # for zoom 5.5

        self.reset_game()

        self.__dict__.update(**kwargs)
        self.frames = [Frame(**f) for f in self.frames]
        # self.frames = {f[0]: Frame(**f[1]) for f in self.frames.items()}

    def reset_game(self):
        self.maze = None
        self.frames = []

    def asdict(self, f_start=-1, f_end=-1):
        if f_end < 0:
            # show all frames by default
            frames_as_dict = [f.asdict() for f in self.frames]
        else:
            frames_as_dict = [f.asdict() for f in self.frames[f_start:f_end]]
        return {
            "game_id": self.game_id,
            "level_seed": self.level_seed,
            "rl_agent_seed": self.rl_agent_seed,
            "zoom": self.zoom,
            "bgzoom": self.bgzoom,
            "world_theme_n": self.world_theme_n,
            "agent_theme_n": self.agent_theme_n,
            "background_themes": self.background_themes,
            "ground_themes": self.ground_themes,
            "agent_themes": self.agent_themes,
            "monster_names": self.monster_names,
            "video_res": self.video_res,
            "maze_w": self.maze_w,
            "maze_h": self.maze_h,
            "maze": self.maze if self.maze is not None else None,
            "frames": frames_as_dict,
        }

    def __repr__(self):
        return json.dumps(self.asdict())

    def save_json(self, json_path, f_start=-1, f_end=-1):
        with open(json_path, "w") as f:
            json.dump(self.asdict(f_start, f_end), f, indent=2)

    def load_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.reset_game()
        self.__dict__.update(**data)
        self.frames = [Frame(**f) for f in self.frames]
        # self.frames = {f[0]: Frame(**f[1]) for f in self.frames.items()}

        self.flatten_monster_names()
        self.reset_eaten_coins()

    def flatten_monster_names(self):
        # the order is important!
        self.flattened_monster_names = self.monster_names['ground']
        self.flattened_monster_names.extend(self.monster_names['walking'])
        self.flattened_monster_names.extend(self.monster_names['flying'])

    # NOTE: some coins might be missing due to how 3s clip json is saved
    # reset all eaten coins to put them back
    # this is a temporary fix until we regenerate all jsons
    def reset_eaten_coins(self):
        for coin_loc in self.frames[-1].coins_eaten:
            # note the game rows are saved as strings
            # NOTE: '1' is the yellow coin, we also has another type '2' that is the red gem
            # but the json with '2' enabled should not have this issue
            if self.maze[coin_loc[1]][coin_loc[0]] == '.':
                self.maze[coin_loc[1]] = self.maze[coin_loc[1]][:coin_loc[0]] + '1' + self.maze[coin_loc[1]][(coin_loc[0] + 1):]


class Frame:
    def __init__(self, **kwargs):
        self.frame_id = -1
        self.file_name = ""
        self.state_time = 0
        self.coins_eaten = []
        self.agent = None
        self.monsters = []

        self.__dict__.update(**kwargs)
        if 'agent' in self.__dict__ and self.agent is not None:
            self.agent = Agent(**self.agent)
        if 'monsters' in self.__dict__:
            self.monsters = [Monster(**m) for m in self.monsters]

    def asdict(self):
        return {
            "frame_id": self.frame_id,
            "file_name": self.file_name,
            "state_time": self.state_time,
            "coins_eaten": self.coins_eaten,
            "agent": self.agent.asdict() if self.agent is not None else None,
            "monsters": [m.asdict() for m in self.monsters]
        }

    def __repr__(self):
        return json.dumps(self.asdict())


class Agent:
    def __init__(
        self, x, y, vx=0.0, vy=0.0, time_alive=0,
        ladder=False, spring=0, is_killed=False,
        killed_animation_frame_cnt=0,
        finished_level_frame_cnt=0,
        killed_monster=False,
        bumped_head=False,
        collected_coin=False,
        collected_gem=False,
        power_up_mode=False,
        **kwargs
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.time_alive = time_alive
        self.ladder = ladder  # for climb pose
        self.spring = spring  # for duck pose

        # states related to agent dying or finishing animations
        self.is_killed = is_killed
        self.killed_animation_frame_cnt = killed_animation_frame_cnt
        self.finished_level_frame_cnt = finished_level_frame_cnt
        self.killed_monster = killed_monster
        self.bumped_head = bumped_head
        self.collected_coin = collected_coin
        self.collected_gem = collected_gem
        self.power_up_mode = power_up_mode

        self.anim_freq = 5    # hard-coded

        # decide whether to flip asset horizontally
        self.is_facing_right = True
        if self.vx < 0:
            self.is_facing_right = False

        # decide which of the two walk/climb asset to use
        self.walk1_mode = True
        if (self.time_alive // self.anim_freq) % 2 != 0:
            self.walk1_mode = False

        self.pose = self.get_pose()

        # kwargs are ignored
        # self.__dict__.update(**kwargs)

    def get_pose(self):
        if self.is_killed:
            return "hit"
        if self.ladder:
            if self.walk1_mode:
                return "climb1"
            else:
                return "climb2"
        if self.vy != 0:
            return "jump"
        if self.spring != 0:
            return "duck"
        if self.vx == 0:
            return "stand"
        if self.walk1_mode:
            return "walk1"
        else:
            return "walk2"

    def asdict(self):
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "time_alive": self.time_alive,
            "ladder": self.ladder,
            "spring": self.spring,
            "is_killed": self.is_killed,
            "killed_animation_frame_cnt": self.killed_animation_frame_cnt,
            "finished_level_frame_cnt": self.finished_level_frame_cnt,
            "killed_monster": self.killed_monster,
            "bumped_head": self.bumped_head,
            "collected_coin": self.collected_coin,
            "collected_gem": self.collected_gem,
            "power_up_mode": self.power_up_mode,
            "anim_freq": self.anim_freq,
            "is_facing_right": self.is_facing_right,
            "walk1_mode": self.walk1_mode,
            "pose": self.pose,
        }

    def __repr__(self):
        return json.dumps(self.asdict())


class Monster:
    def __init__(
        self, m_id, x, y, vx=0.0, vy=0.0, theme=0,
        is_flying=False, is_walking=False, is_jumping=False, is_dead=False,
        time=0, anim_freq=1, monster_dying_frame_cnt=0, **kwargs
    ):
        self.m_id = m_id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.theme = theme   # monster type (saw, snail, slime, etc.)
        self.is_flying = is_flying
        self.is_walking = is_walking
        self.is_jumping = is_jumping
        self.is_dead = is_dead
        self.time = time
        self.anim_freq = anim_freq
        self.monster_dying_frame_cnt = monster_dying_frame_cnt

        # decide which of the two walk/climb asset to use
        self.walk1_mode = True
        if self.is_jumping:
            # for jumping monster, walk1 asset is decided by vertical speed
            if self.vy != 0:
                self.walk1_mode = False
        elif (self.time // self.anim_freq) % 2 != 0:
            self.walk1_mode = False

        # kwargs are ignored
        # self.__dict__.update(**kwargs)

    def asdict(self):
        return {
            "m_id": self.m_id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "theme": self.theme,
            "is_flying": self.is_flying,
            "is_walking": self.is_walking,
            "is_jumping": self.is_jumping,
            "is_dead": self.is_dead,
            "time": self.time,
            "anim_freq": self.anim_freq,
            "monster_dying_frame_cnt": self.monster_dying_frame_cnt,
            "walk1_mode": self.walk1_mode,
        }

    def __repr__(self):
        return json.dumps(self.asdict())
