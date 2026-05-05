#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from PIL import Image

U_MAZE = "#####\\#GOO#\\###O#\\#OOO#\\#####"
LARGE_MAZE = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
MEDIUM_MAZE = '########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########'

MAZE_SPECS = {
    'U_MAZE': U_MAZE,
    'LARGE_MAZE': LARGE_MAZE,
    'MEDIUM_MAZE': MEDIUM_MAZE,
}


# Maze walls at (w+1,h+1) size [0.5,0.5,0.2] → maze spans [0.5, 5.5] in x,y.
# Four regions outside that box: (x_lo,x_hi), (y_lo,y_hi) for top/bottom left/right.
_MAZE_MAX = 5.5
_PAD = 0.6
DISTRACTOR_REGIONS = [
    ("top_right", (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD), (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD)),
    ("top_left", (-_PAD, 0.5 - 0.1), (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD)),
    ("bottom_right", (_MAZE_MAX + 0.1, _MAZE_MAX + _PAD), (-_PAD, 0.5 - 0.1)),
    ("bottom_left", (-_PAD, 0.5 - 0.1), (-_PAD, 0.5 - 0.1)),
]
DISTRACTOR_Z = 0.02


def _sample_distractor_xy(rng):
    """Pick one of four outside regions (TL/TR/BL/BR) and sample (x,y) in it."""
    idx = int(rng.integers(0, 4))
    _, (x_lo, x_hi), (y_lo, y_hi) = DISTRACTOR_REGIONS[idx]
    x = float(rng.uniform(x_lo, x_hi))
    y = float(rng.uniform(y_lo, y_hi))
    return x, y


def _sample_distractor_rgba(rng):
    """Random color, avoiding black and avoiding predominantly red/green."""
    for _ in range(50):
        rgb = rng.uniform(0.35, 0.95, size=3)
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        predominantly_red = r > g and r > b and r > 0.5
        predominantly_green = g > r and g > b and g > 0.5
        if not predominantly_red and not predominantly_green:
            return r, g, b, 1.0
    return 0.85, 0.2, 0.75, 1.0  # fallback: blue-violet


def render_maze_sample(
    maze_spec,
    init_state,
    goal_state,
    background_builtin="gradient",
    background_rgb1="0.18 0.05 0.35",
    background_rgb2="0.5 0.22 0.55",
    render_size=224,
    distractor_seed=None,
):
    env = PointMazeWrapper(
        maze_spec=maze_spec,
        return_value="state",
        background_builtin=background_builtin,
        background_rgb1=background_rgb1,
        background_rgb2=background_rgb2,
        with_target=True
    )
    
    env.return_value = 'obs'
    env.prepare_for_render()
    
    qpos = init_state[:2]
    qvel = init_state[2:4] if len(init_state) >= 4 else np.array([0.0, 0.0])
    env.set_state(qpos, qvel)
    
    goal_pos_world = goal_state[:2]
    env.set_target(goal_pos_world)
    env.set_marker()

    if distractor_seed is not None:
        distractor_site_id = env.model.site_name2id('distractor_site')
        rng = np.random.default_rng(distractor_seed)
        x, y = _sample_distractor_xy(rng)
        env.sim.data.site_xpos[distractor_site_id] = np.array([x, y, DISTRACTOR_Z])
        r, g, b, a = _sample_distractor_rgba(rng)
        env.model.site_rgba[distractor_site_id] = np.array([r, g, b, a])
    
    
    img = env.sim.render(render_size, render_size)
    
    env.close()
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Generate point maze environment samples with different start and goal positions'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of maze samples to generate (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='observations',
        help='Output directory for generated samples (default: observations)'
    )
    parser.add_argument(
        '--maze-spec',
        type=str,
        default='U_MAZE',
        choices=list(MAZE_SPECS.keys()),
        help='Maze specification to use (default: U_MAZE)'
    )
    parser.add_argument(
        '--background-builtin',
        type=str,
        default='gradient',
        help='Background texture type: "gradient" or "checker" (default: gradient)'
    )
    parser.add_argument(
        '--background-rgb1',
        type=str,
        default='0.18 0.05 0.35',
        help='Background gradient end 1 – dark violet (default)'
    )
    parser.add_argument(
        '--background-rgb2',
        type=str,
        default='0.5 0.22 0.55',
        help='Background gradient end 2 – lighter purple/magenta (default)'
    )
    parser.add_argument(
        '--render-size',
        type=int,
        default=224,
        help='Size of rendered images (default: 224)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-distractor',
        action='store_true',
        help='Disable the black-dot distractor geom'
    )
    parser.add_argument(
        '--distractor-chunk-size',
        type=int,
        default=1,
        metavar='N',
        help='Use same distractor position for N consecutive samples (default: 1)'
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    maze_spec = MAZE_SPECS[args.maze_spec]
    
    print(f"Generating {args.num_samples} maze samples...")
    print(f"Maze spec: {args.maze_spec}")
    print(f"Output directory: {output_dir}")
    
    temp_env = PointMazeWrapper(
        maze_spec=maze_spec,
        return_value="state",
        background_builtin=args.background_builtin,
        background_rgb1=args.background_rgb1,
        background_rgb2=args.background_rgb2,
        with_target=True
    )
    
    distractor_seed_fn = (lambda i: None) if args.no_distractor else (
        lambda i: args.seed + (i // args.distractor_chunk_size)
    )
    for i in range(args.num_samples):
        init_state, goal_state = temp_env.sample_random_init_goal_states(seed=args.seed + i)
        dseed = distractor_seed_fn(i)

        img = render_maze_sample(
            maze_spec=maze_spec,
            init_state=init_state,
            goal_state=goal_state,
            background_builtin=args.background_builtin,
            background_rgb1=args.background_rgb1,
            background_rgb2=args.background_rgb2,
            render_size=args.render_size,
            distractor_seed=dseed,
        )
        
        img_pil = Image.fromarray(img)
        img_path = output_dir / f"maze_sample_{i:03d}.png"
        img_pil.save(img_path)
        
        print(f"  Generated sample {i+1}/{args.num_samples}: {img_path}")
    
    temp_env.close()
    
    print(f"\nAll samples saved to: {output_dir}")


if __name__ == '__main__':
    main()
