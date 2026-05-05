import os
import numpy as np
import gym
from env.pointmaze.maze_model import MazeEnv
from utils import aggregate_dct
from datasets.domain_randomization import (
    DISTRACTOR_Z,
)

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])

# U_MAZE movable area (matches sample_random_init_goal_states validity). Inset keeps
# distractors off walls. Format per region: (x_lo, x_hi, y_lo, y_hi).
_MOVABLE_INSET = 0.14  # distance inside boundaries so distractor is never on wall
_U_MAZE_INSET_REGIONS = [
    (0.5 + _MOVABLE_INSET, 1.1 - _MOVABLE_INSET, 0.5 + _MOVABLE_INSET, 3.1 - _MOVABLE_INSET),   # left arm
    (2.5 + _MOVABLE_INSET, 3.1 - _MOVABLE_INSET, 0.5 + _MOVABLE_INSET, 3.1 - _MOVABLE_INSET),   # right arm
    (1.1 + _MOVABLE_INSET, 2.5 - _MOVABLE_INSET, 2.5 + _MOVABLE_INSET, 3.1 - _MOVABLE_INSET),   # bottom
]


def _clamp_to_u_maze_inset(x, y):
    """Clamp (x,y) to U_MAZE movable area with inset. Returns (x', y') inside floor only."""
    # Which region does (x,y) belong to (or is nearest)?
    best_x, best_y = x, y
    best_d2 = np.inf
    for (x_lo, x_hi, y_lo, y_hi) in _U_MAZE_INSET_REGIONS:
        xc = np.clip(x, x_lo, x_hi)
        yc = np.clip(y, y_lo, y_hi)
        d2 = (x - xc) ** 2 + (y - yc) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_x, best_y = xc, yc
    return float(best_x), float(best_y)


class PointMazeWrapper(MazeEnv):
    def __init__(self, background='default', with_distractor=False, 
                 distractor_pos=None, distractor_rgba=None, **kwargs):
        """
        Args:
            background: Background configuration (string or dict)
            with_distractor: Whether to add a distractor (bool)
            distractor_pos: Optional (x, y) tuple for distractor position. 
                          If None and with_distractor=True, samples randomly from one of 4 regions
            distractor_rgba: Optional (r, g, b, a) tuple for distractor color.
                           If None and with_distractor=True, defaults to black (0.0, 0.0, 0.0, 1.0)
        """
        background_configs = {
            'default': {
                'background_builtin': 'checker',
                'background_rgb1': '0.2 0.3 0.4',
                'background_rgb2': '0.1 0.2 0.3'
            },
            'slight_change': {
                'background_builtin': 'checker',
                'background_rgb1': '0.4 0.5 0.6',
                'background_rgb2': '0.3 0.4 0.5'
            },
            'gradient': {
                'background_builtin': 'gradient',
                'background_rgb1': '0.2 0.3 0.4',
                'background_rgb2': '0.1 0.2 0.3'
            },
            'gradient_aggressive': {
                'background_builtin': 'gradient',
                'background_rgb1': '0.18 0.05 0.35',
                'background_rgb2': '0.5 0.22 0.55'
            }
        }
        if isinstance(background, dict):
            required = {'background_builtin', 'background_rgb1', 'background_rgb2'}
            if required.issubset(background.keys()):
                kwargs.update({k: background[k] for k in required})
            else:
                raise ValueError(
                    f"Custom background dict must contain: {required}. Got keys: {list(background.keys())}"
                )
        elif background in background_configs:
            kwargs.update(background_configs[background])
        else:
            raise ValueError(
                f"Unknown background: {background}. Must be one of {list(background_configs.keys())} "
                "or a dict with keys background_builtin, background_rgb1, background_rgb2."
            )
        
        self.with_distractor = with_distractor
        self._distractor_site_names = ['distractor_site', 'distractor_site_2']
        if with_distractor:
            if distractor_pos is None:
                self.distractor_circle_center = None  # two fixed random distractors, no circle
                self.distractor_pos = None  # filled after super().__init__ with two sampled positions
                self.distractor_rgba = None  # filled with two distinct default colors
            else:
                self.distractor_circle_center = None
                # Normalize to list of 2 positions: (x,y) -> [(x,y),(x,y)], [(x1,y1),(x2,y2)] -> as-is
                try:
                    pos_list = list(distractor_pos)
                    if len(pos_list) != 2 or not all(len(p) == 2 for p in pos_list):
                        pos_list = [tuple(distractor_pos), tuple(distractor_pos)]
                except (TypeError, ValueError):
                    pos_list = [tuple(distractor_pos), tuple(distractor_pos)]
                self.distractor_pos = [tuple(p) for p in pos_list]
                # Normalize to list of 2 rgbas: (r,g,b,a) -> both get it; [rgba1,rgba2] -> as-is
                if distractor_rgba is not None and len(distractor_rgba) == 2 and len(distractor_rgba[0]) == 4:
                    self.distractor_rgba = [tuple(distractor_rgba[0]), tuple(distractor_rgba[1])]
                else:
                    one = tuple(distractor_rgba) if distractor_rgba is not None else (1.0, 1.0, 0.0, 1.0)
                    self.distractor_rgba = [one, (0.85, 0.2, 0.75, 1.0)]
        else:
            self.distractor_pos = None
            self.distractor_rgba = None
            self.distractor_circle_center = None
        
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
        
        # When two distractors and positions not provided, sample two positions and two colors after model exists
        if self.with_distractor and self.distractor_pos is None:
            base = (self._seed + 100) if getattr(self, '_seed', None) is not None else None
            self.distractor_pos = [
                self.sample_distractor_inside_maze(seed=base),
                self.sample_distractor_inside_maze(seed=base + 1 if base is not None else None),
            ]
            self.distractor_rgba = [(1.0, 1.0, 0.0, 1.0), (0.85, 0.2, 0.75, 1.0)]  # yellow, purple
        
        if self.with_distractor:
            self._add_distractor()
    
    def _add_distractor(self):
        """Add two distractors (different colors, random positions in movable area) if with_distractor is True."""
        if not hasattr(self, 'with_distractor') or not self.with_distractor or self.distractor_pos is None:
            try:
                if hasattr(self, 'model') and self.model is not None:
                    for name in getattr(self, '_distractor_site_names', ['distractor_site', 'distractor_site_2']):
                        sid = self.model.site_name2id(name)
                        self.model.site_rgba[sid] = np.array([0, 0, 0, 0])
            except Exception:
                pass
            return
        if not hasattr(self, 'sim') or not hasattr(self, 'model') or self.sim is None or self.model is None:
            return
        try:
            for i, site_name in enumerate(getattr(self, '_distractor_site_names', ['distractor_site', 'distractor_site_2'])):
                sid = self.model.site_name2id(site_name)
                x, y = self.distractor_pos[i]  # state coords; site uses world = state+1 (like target_site)
                self.sim.data.site_xpos[sid] = np.array([x + 1, y + 1, DISTRACTOR_Z])
                r, g, b, a = self.distractor_rgba[i]
                self.model.site_rgba[sid] = np.array([r, g, b, 1.0])
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to set distractor positions: {e}")
            import traceback
            traceback.print_exc()
    
    def reset(self):
        """Override reset to add distractor if enabled."""
        obs, state = super().reset()
        # Reset step counters for circular motion and center walk
        if self.with_distractor and self.distractor_circle_center is not None:
            self.distractor_step_count = 0
            self.distractor_center_walk_step = 0
            # Re-initialize center position on reset
            if hasattr(self, 'distractor_center_bounds'):
                cx = self.distractor_rng.uniform(
                    self.distractor_center_bounds['x_min'],
                    self.distractor_center_bounds['x_max']
                )
                cy = self.distractor_rng.uniform(
                    self.distractor_center_bounds['y_min'],
                    self.distractor_center_bounds['y_max']
                )
                self.distractor_circle_center = (cx, cy)
        self._add_distractor()
        return obs, state
    
    def step(self, action):
        """Override step to maintain distractor visibility."""
        obs, reward, done, info = super().step(action)
        # Increment step counter for circular motion
        if self.with_distractor and self.distractor_circle_center is not None:
            self.distractor_step_count += 1
            
            # Update circle center position with random walk
            if hasattr(self, 'distractor_center_walk_interval'):
                self.distractor_center_walk_step += 1
                if self.distractor_center_walk_step >= self.distractor_center_walk_interval:
                    self.distractor_center_walk_step = 0
                    # Random walk: add small random offset to center
                    # Bias towards right (positive x) and down (positive y) since starting in top-left
                    cx, cy = self.distractor_circle_center
                    # Bias: 70% chance to move right/down, 30% chance to move left/up
                    # For x: positive (right) is more likely
                    dx_sign = 1.0 if self.distractor_rng.random() < 0.7 else -1.0
                    # For y: positive (down) is more likely  
                    dy_sign = 1.0 if self.distractor_rng.random() < 0.7 else -1.0
                    dx = self.distractor_rng.uniform(0, self.distractor_center_walk_step_size) * dx_sign
                    dy = self.distractor_rng.uniform(0, self.distractor_center_walk_step_size) * dy_sign
                    
                    # Clamp to bounds to stay outside maze
                    new_cx = np.clip(
                        cx + dx,
                        self.distractor_center_bounds['x_min'],
                        self.distractor_center_bounds['x_max']
                    )
                    new_cy = np.clip(
                        cy + dy,
                        self.distractor_center_bounds['y_min'],
                        self.distractor_center_bounds['y_max']
                    )
                    self.distractor_circle_center = (new_cx, new_cy)
        
        self._add_distractor()
        return obs, reward, done, info
    
    def _render_frame(self):
        """Override to set distractor positions right before rendering (two distractors)."""
        if self.with_distractor and self.distractor_pos is not None and len(self.distractor_pos) == 2:
            try:
                for i, site_name in enumerate(getattr(self, '_distractor_site_names', ['distractor_site', 'distractor_site_2'])):
                    sid = self.model.site_name2id(site_name)
                    x, y = _clamp_to_u_maze_inset(*self.distractor_pos[i])
                    wx, wy = x + 1, y + 1
                    try:
                        self.model.site_pos[sid] = np.array([wx, wy, DISTRACTOR_Z])
                    except (AttributeError, ValueError, TypeError):
                        pass
                    self.sim.data.site_xpos[sid] = np.array([wx, wy, DISTRACTOR_Z])
                    r, g, b, a = self.distractor_rgba[i]
                    self.model.site_rgba[sid] = np.array([r, g, b, 1.0])
            except Exception as e:
                import random
                if random.random() < 0.01:
                    import warnings
                    warnings.warn(f"Failed to set distractors in _render_frame: {e}")
        return super()._render_frame()
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)

        def generate_state():
            valid = False
            while not valid:
                x = rs.uniform(0.5, 3.1)
                y = rs.uniform(0.5, 3.1)
                valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1))\
                        or ((1.1 < x < 2.5) and (2.5 <= y <= 3.1))
            state = np.array([
                x, 
                y,
                rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
            ])
            return state

        init_state = generate_state()
        goal_state = generate_state()
        return init_state, goal_state

    def sample_distractor_inside_maze(self, seed=None):
        """
        Sample (x, y) inside the maze's movable area, inset from walls so the
        distractor is never on a wall. Uses U_MAZE geometry.
        Returns (x, y) in world coords.
        """
        rng = np.random.default_rng(seed) if seed is not None else getattr(
            self, 'distractor_rng', None
        ) or np.random.default_rng()
        idx = int(rng.integers(0, len(_U_MAZE_INSET_REGIONS)))
        x_lo, x_hi, y_lo, y_hi = _U_MAZE_INSET_REGIONS[idx]
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        return x, y

    def set_distractor_inside_maze(self, seed=None):
        """
        Place both distractors at random positions inside the maze's movable area
        (inset from walls), with different default colors. Call after __init__ or
        reset if with_distractor=True.
        """
        if not getattr(self, 'with_distractor', False):
            return
        base = seed
        self.distractor_pos = [
            self.sample_distractor_inside_maze(seed=base),
            self.sample_distractor_inside_maze(seed=base + 1 if base is not None else None),
        ]
        self.distractor_rgba = [(1.0, 1.0, 0.0, 1.0), (0.85, 0.2, 0.75, 1.0)]  # yellow, purple
        self.distractor_circle_center = None
        self._add_distractor()
    
    def update_env(self, env_info):
        pass 
    
    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
