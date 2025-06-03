from gym.envs.registration import register
from .pointmaze import U_MAZE
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id='point_maze',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)

register(
    id='point_maze_slight_change',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
        # MOD1 scalable background
        'background_builtin':"checker",
        'background_rgb1':"0.4 0.5 0.6",
        'background_rgb2':"0.3 0.4 0.5"
    }
)
register(
    id='point_maze_gradient',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
        # MOD1 scalable background
        'background_builtin':"gradient",
        'background_rgb1':"0.2 0.3 0.4",
        'background_rgb2':"0.1 0.2 0.3"
    }
)
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)