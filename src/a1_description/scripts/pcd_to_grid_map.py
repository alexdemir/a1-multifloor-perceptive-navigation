#!/usr/bin/env python3
"""
3-Story PCD to Nav2 2.5D Multi-Floor Occupancy Grid Map Converter.

Converts a 3-story DLIO 3D point cloud (dlio_map.pcd) into aligned ROS 2 Nav2 maps:
  1. floor1_map.{pgm,yaml}: Floor 1 (z_world=0.0m) + ramp_1_2 corridor
  2. floor2_map.{pgm,yaml}: Floor 2 (z_world=1.5m) + ramp_1_2 (shared) + ramp_2_3 entry
  3. floor3_map.{pgm,yaml}: Floor 3 (z_world=3.0m) + ramp_2_3 (shared) + bottom landing

Supports `--from-world` to synthesize exact baseline maps directly from the 3-story
building CAD geometry before a fresh 3-story PCD recording is captured.
"""

import argparse
import os
import sys
import math
import numpy as np
import cv2


def load_pcd_binary(pcd_path):
    dt = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<f4'), ('t', '<u4'), ('time', '<f4'), ('timestamp', '<f8')
    ])
    with open(pcd_path, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8', errors='ignore')
            if line.startswith('DATA'):
                break
        data = np.fromfile(f, dtype=dt)
    return data['x'].copy(), data['y'].copy(), data['z'].copy()


def align_pcd_to_world(x0, y0, z0):
    w = (z0 > 0.10) & (z0 < 0.80)
    wx, wy = x0[w], y0[w]

    north = wy > (np.percentile(wy, 98) - 0.35)
    south = wy < (np.percentile(wy, 2) + 0.35)
    slope_n = np.polyfit(wx[north], wy[north], 1)[0]
    slope_s = np.polyfit(wx[south], wy[south], 1)[0]
    yaw_err = math.atan(0.5 * (slope_n + slope_s))

    c, s = math.cos(-yaw_err), math.sin(-yaw_err)
    xr = c * x0 - s * y0
    yr = s * x0 + c * y0

    wx2, wy2 = xr[w], yr[w]
    x_min_wall = np.median(wx2[wx2 < (np.percentile(wx2, 2) + 0.35)])
    x_max_wall = np.median(wx2[wx2 > (np.percentile(wx2, 98) - 0.35)])
    y_min_wall = np.median(wy2[wy2 < (np.percentile(wy2, 2) + 0.35)])
    y_max_wall = np.median(wy2[wy2 > (np.percentile(wy2, 98) - 0.35)])

    cx = 0.5 * (x_min_wall + x_max_wall)
    cy = 0.5 * (y_min_wall + y_max_wall)

    x = xr - cx
    y = yr - cy
    print(f"Aligned PCD -> World: yaw_corr={math.degrees(-yaw_err):+.2f} deg, shift=({-cx:+.3f}, {-cy:+.3f}) m")
    return x, y, z0


def save_nav2_map(grid, min_x, min_y, resolution, map_name, output_dir):
    ros_pgm = np.flipud(grid)
    pgm_filename = f"{map_name}.pgm"
    yaml_filename = f"{map_name}.yaml"
    cv2.imwrite(os.path.join(output_dir, pgm_filename), ros_pgm)
    yaml_path = os.path.join(output_dir, yaml_filename)
    with open(yaml_path, 'w') as f:
        f.write(
            f"image: {pgm_filename}\n"
            f"mode: trinary\n"
            f"resolution: {resolution}\n"
            f"origin: [{min_x:.3f}, {min_y:.3f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.25\n"
        )
    print(f"Saved: {yaml_path} ({grid.shape[1]}x{grid.shape[0]} px, res={resolution}m)")


def draw_rects(grid, rects, val, min_x, min_y, resolution):
    h, w = grid.shape
    for (x1, y1, x2, y2) in rects:
        c1 = max(0, int(round((x1 - min_x) / resolution)))
        c2 = min(w - 1, int(round((x2 - min_x) / resolution)))
        r1 = max(0, int(round((y1 - min_y) / resolution)))
        r2 = min(h - 1, int(round((y2 - min_y) / resolution)))
        grid[r1:r2 + 1, c1:c2 + 1] = val


def rasterize_map(x, y, ground_mask, obs_mask, min_x, min_y, width, height, resolution,
                  extra_free_rects=None, extra_obs_rects=None):
    grid = np.full((height, width), 205, dtype=np.uint8)

    if x is not None and ground_mask is not None and np.any(ground_mask):
        gx, gy = x[ground_mask], y[ground_mask]
        g_cols = np.clip(((gx - min_x) / resolution).astype(int), 0, width - 1)
        g_rows = np.clip(((gy - min_y) / resolution).astype(int), 0, height - 1)
        grid[g_rows, g_cols] = 254

        free_bin = (grid == 254).astype(np.uint8)
        kernel_free = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        free_bin = cv2.dilate(free_bin, kernel_free, iterations=2)
        free_bin = cv2.morphologyEx(free_bin, cv2.MORPH_CLOSE, kernel_free, iterations=2)
        grid[free_bin == 1] = 254

    if extra_free_rects:
        draw_rects(grid, extra_free_rects, 254, min_x, min_y, resolution)

    if x is not None and obs_mask is not None and np.any(obs_mask):
        ox, oy = x[obs_mask], y[obs_mask]
        o_cols = np.clip(((ox - min_x) / resolution).astype(int), 0, width - 1)
        o_rows = np.clip(((oy - min_y) / resolution).astype(int), 0, height - 1)
        obs_bin = np.zeros((height, width), dtype=np.uint8)
        obs_bin[o_rows, o_cols] = 1
        kernel_obs = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        obs_bin = cv2.dilate(obs_bin, kernel_obs, iterations=1)
        grid[obs_bin == 1] = 0

    if extra_obs_rects:
        draw_rects(grid, extra_obs_rects, 0, min_x, min_y, resolution)

    return grid


def generate_from_world(maps_dir, min_x, min_y, width, height, res):
    """Generates exact 3-floor Nav2 maps directly from multi_floor_building.world CAD geometry."""
    outer_walls = [
        (-7.10,  5.90,  7.10,  6.10),  # North
        (-7.10, -6.10,  7.10, -5.90),  # South
        (-7.10, -6.10, -6.90,  6.10),  # West
        ( 6.90, -6.10,  7.10,  6.10),  # East
    ]

    # Floor 1: Full interior + ramp_1_2 guard walls + room_divider_1
    f1_free = [(-6.90, -5.90, 6.90, 5.90)]
    f1_obs = outer_walls + [
        (-2.60, -0.50, -2.40,  2.30),  # room_divider_1
        (-5.55, -2.78,  0.55, -2.62),  # ramp_1_2 north guard/under wall
        (-5.55, -4.98,  0.55, -4.82),  # ramp_1_2 south guard wall
        ( 0.45, -4.90,  0.58, -2.70),  # Virtual F1/F2 gate at ramp_1_2 top
    ]
    g1 = rasterize_map(None, None, None, None, min_x, min_y, width, height, res, f1_free, f1_obs)
    save_nav2_map(g1, min_x, min_y, res, "floor1_map", maps_dir)

    # Floor 2: East deck [0.5, 6.9] + shared ramp_1_2 [-6.8..1.5, -5.2..-2.85]
    f2_free = [
        ( 0.50, -5.90,  6.90,  5.90),  # Floor 2 main deck
        (-6.80, -5.20,  1.50, -2.85),  # ramp_1_2 bottom landing + corridor
    ]
    f2_obs = outer_walls + [
        ( 0.43, -2.70,  0.58,  2.70),  # Balcony guard mid
        ( 0.43,  4.90,  0.58,  5.95),  # Balcony guard north
        ( 0.43, -5.95,  0.58, -4.90),  # Balcony guard south
        ( 3.60, -2.30,  3.80,  2.30),  # second_floor_divider
        (-5.55, -2.78,  0.55, -2.62),  # ramp_1_2 north guard
        (-5.55, -4.98,  0.55, -4.82),  # ramp_1_2 south guard
        ( 0.43,  2.70,  0.58,  4.90),  # Virtual F2/F3 gate at ramp_2_3 start
    ]
    g2 = rasterize_map(None, None, None, None, min_x, min_y, width, height, res, f2_free, f2_obs)
    save_nav2_map(g2, min_x, min_y, res, "floor2_map", maps_dir)

    # Floor 3: West deck [-6.9..-0.5, -5.9..2.7] + arrival [-6.9..-4.5, 2.7..5.9] + shared ramp_2_3 [-4.5..3.0, 2.85..4.75]
    f3_free = [
        (-6.90, -5.90, -0.50,  2.70),  # Floor 3 main west deck
        (-6.90,  2.70, -4.50,  5.90),  # Floor 3 ramp arrival deck
        (-4.60,  2.85,  3.00,  4.75),  # ramp_2_3 corridor + Floor 2 bottom handoff landing
    ]
    f3_obs = outer_walls + [
        (-0.58, -5.95, -0.43,  2.78),  # f3_guard_east
        (-4.55,  2.62, -0.43,  2.78),  # f3_guard_north + ramp_2_3 south guard
        (-4.58,  4.82, -4.43,  5.95),  # f3_guard_landing_ne
        (-4.55,  2.62,  1.55,  2.78),  # ramp_2_3 south guard
        (-4.55,  4.82,  1.55,  4.98),  # ramp_2_3 north guard
        (-3.80, -3.90, -3.60,  0.30),  # third_floor_divider
    ]
    g3 = rasterize_map(None, None, None, None, min_x, min_y, width, height, res, f3_free, f3_obs)
    save_nav2_map(g3, min_x, min_y, res, "floor3_map", maps_dir)
    return g1, g2, g3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-world', action='store_true',
                        help='Generate exact 3-floor maps from world CAD geometry (no PCD required)')
    parser.add_argument('--pcd', default='/home/alex/a1_navigation_ws/src/a1_description/maps/dlio_map.pcd')
    parser.add_argument('--out', default='/home/alex/a1_navigation_ws/src/a1_description/maps')
    args = parser.parse_args()

    res = 0.05
    min_x, max_x = -8.0, 8.0
    min_y, max_y = -7.0, 7.0
    width = int(math.ceil((max_x - min_x) / res))
    height = int(math.ceil((max_y - min_y) / res))

    if args.from_world or not os.path.exists(args.pcd):
        print("Generating 3-floor maps from multi_floor_building.world CAD geometry...")
        g1, g2, g3 = generate_from_world(args.out, min_x, min_y, width, height, res)
    else:
        x0, y0, z = load_pcd_binary(args.pcd)
        print(f"Loaded {len(z)} points from {args.pcd} (Z in [{z.min():.2f}, {z.max():.2f}]m)")
        if z.max() < 2.5:
            print("Note: PCD does not contain Floor 3 points (z_max < 2.5m). Using --from-world fallback for Floor 3.")
            g1, g2, g3 = generate_from_world(args.out, min_x, min_y, width, height, res)
        else:
            x, y, z = align_pcd_to_world(x0, y0, z)

            # Ramp 1->2: x in [-5.5, 0.5], y in [-4.9, -2.7], z_dlio: -0.30 -> +1.20
            is_r12_box = (x >= -5.60) & (x <= 0.60) & (y >= -5.05) & (y <= -2.55)
            is_r12_int = (x >= -5.55) & (x <= 0.55) & (y >= -4.75) & (y <= -2.85)
            z_r12 = -0.30 + 0.25 * (x + 5.50)
            r12_gnd = is_r12_int & (np.abs(z - z_r12) <= 0.22)
            r12_obs = is_r12_box & (~is_r12_int) & ((z - z_r12) >= 0.15) & ((z - z_r12) <= 1.15)

            # Ramp 2->3: x in [-4.5, 1.5], y in [2.7, 4.9], z_dlio: +1.20 (x=1.5) -> +2.70 (x=-4.5)
            is_r23_box = (x >= -4.60) & (x <= 1.60) & (y >= 2.55) & (y <= 5.05)
            is_r23_int = (x >= -4.55) & (x <= 1.55) & (y >= 2.85) & (y <= 4.75)
            z_r23 = 1.20 + 0.25 * (1.50 - x)
            r23_gnd = is_r23_int & (np.abs(z - z_r23) <= 0.22)
            r23_obs = is_r23_box & (~is_r23_int) & ((z - z_r23) >= 0.15) & ((z - z_r23) <= 1.15)

            # 1. Floor 1
            f1_gnd = ((~is_r12_box) & (z >= -0.45) & (z <= -0.15)) | r12_gnd
            f1_obs = ((~is_r12_box) & (z >= -0.08) & (z <= 0.85)) | r12_obs
            g1 = rasterize_map(
                x, y, f1_gnd, f1_obs, min_x, min_y, width, height, res,
                extra_free_rects=[(-6.40, -4.70, 0.40, -2.90)],
                extra_obs_rects=[(-5.55, -2.78, 0.55, -2.62), (-5.55, -4.98, 0.55, -4.82), (0.45, -4.90, 0.58, -2.70)],
            )
            save_nav2_map(g1, min_x, min_y, res, "floor1_map", args.out)

            # 2. Floor 2
            is_f2 = x >= 0.40
            f2_gnd = (is_f2 & (z >= 1.00) & (z <= 1.35)) | r12_gnd
            f2_obs = (is_f2 & (z >= 1.40) & (z <= 2.40)) | r12_obs
            g2 = rasterize_map(
                x, y, f2_gnd, f2_obs, min_x, min_y, width, height, res,
                extra_free_rects=[(-6.80, -5.20, 1.50, -2.85)],
                extra_obs_rects=[
                    (-5.55, -2.78, 0.55, -2.62), (-5.55, -4.98, 0.55, -4.82),
                    (-6.95, -5.40, -6.85, -2.50), (0.43, -2.70, 0.58, 2.70),
                    (0.43, 4.90, 0.58, 5.95), (0.43, -5.95, 0.58, -4.90),
                    (0.43, 2.70, 0.58, 4.90),
                ],
            )
            save_nav2_map(g2, min_x, min_y, res, "floor2_map", args.out)

            # 3. Floor 3
            is_f3 = (x <= -0.40) & ((y <= 2.80) | (x <= -4.40))
            f3_gnd = (is_f3 & (z >= 2.50) & (z <= 2.85)) | r23_gnd
            f3_obs = (is_f3 & (z >= 2.90) & (z <= 3.95)) | r23_obs
            g3 = rasterize_map(
                x, y, f3_gnd, f3_obs, min_x, min_y, width, height, res,
                extra_free_rects=[(-4.60, 2.85, 3.00, 4.75)],
                extra_obs_rects=[
                    (-0.58, -5.95, -0.43, 2.78), (-4.55, 2.62, 1.55, 2.78),
                    (-4.55, 4.82, 1.55, 4.98), (-4.58, 4.82, -4.43, 5.95),
                ],
            )
            save_nav2_map(g3, min_x, min_y, res, "floor3_map", args.out)

    # Verify key waypoints across all 3 floors
    print("\n3-Floor Waypoint Verification:")
    for desc, px, py, expected in [
        ("F1 Spawn",          0.00,  0.00, (254, 205, 205)),
        ("Ramp 1->2 Bottom", -6.05, -3.80, (254, 254, 254)),
        ("Ramp 1->2 Top",     1.20, -3.80, (254, 254, 205)),
        ("Ramp 2->3 Bottom",  2.15,  3.80, (254, 254, 254)),
        ("Ramp 2->3 Top",    -5.30,  3.80, (254, 205, 254)),
        ("F3 Main Room",     -2.00, -1.50, (254, 205, 254)),
    ]:
        c = int((px - min_x) / res)
        r = int((py - min_y) / res)
        vals = (int(g1[r, c]), int(g2[r, c]), int(g3[r, c]))
        print(f"  {desc:18s} ({px:+5.2f}, {py:+5.2f}) -> F1={vals[0]:3d} F2={vals[1]:3d} F3={vals[2]:3d}")


if __name__ == '__main__':
    main()
