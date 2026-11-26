#!/usr/bin/env python3
"""
Min-Max Elevation (Delta-Z Verticality) 3D LiDAR Obstacle Filter.

Filters /ouster/points in lidar_link frame using grid-based height-difference
segmentation (Min-Max Ground Removal):
  1. Pre-crops ceiling points (z > max_z) and robot self-hits (r < min_range).
  2. Bins remaining points into 2D (x, y) vertical columns of size cell_size (0.15m).
  3. Computes delta_z = z_max - z_min per column:
     - Flat ground (even under +/-10 deg trotting roll/pitch): delta_z ~ 0.03m -> rejected
     - 14-deg ramp surface: delta_z ~ 0.04m -> rejected
     - Vertical walls, railings, boxes: delta_z ~ 0.50-0.80m -> kept
  4. Retains walls up to max_range (12.0m) so Nav2 local_costmap raytrace clearing
     continuously sweeps empty floor space.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2


class LidarObstacleFilter(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_filter')

        self.declare_parameter('input_topic', '/ouster/points')
        self.declare_parameter('output_topic', '/ouster/points_filtered')
        self.declare_parameter('min_z', -0.60)
        self.declare_parameter('max_z', 0.45)
        self.declare_parameter('min_range', 0.38)
        self.declare_parameter('max_range', 12.0)
        self.declare_parameter('cell_size', 0.15)
        self.declare_parameter('min_height_diff', 0.22)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.min_r2 = self.min_range ** 2
        self.max_r2 = self.max_range ** 2
        self.cell_size = float(self.get_parameter('cell_size').value)
        self.min_height_diff = float(self.get_parameter('min_height_diff').value)
        self.grid_dim = int(math.ceil((2.0 * self.max_range) / self.cell_size)) + 2

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub = self.create_subscription(PointCloud2, input_topic, self.cloud_cb, sub_qos)
        self.pub = self.create_publisher(PointCloud2, output_topic, pub_qos)
        self._cached_dtype = None
        self._cached_step = None

        self.get_logger().info(
            f'Min-Max Delta-Z LiDAR filter ready: {input_topic} -> {output_topic} '
            f'(cell={self.cell_size:.2f}m, dz>={self.min_height_diff:.2f}m, '
            f'z in [{self.min_z:.2f}, {self.max_z:.2f}]m, r in [{self.min_range:.2f}, {self.max_range:.1f}]m)'
        )

    def _get_xyz_offsets(self, msg: PointCloud2):
        offsets = {}
        for f in msg.fields:
            if f.name in ('x', 'y', 'z'):
                offsets[f.name] = f.offset
        return offsets

    def _publish_filtered(self, msg: PointCloud2, filtered_raw: np.ndarray):
        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(filtered_raw.shape[0])
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = out.width * out.point_step
        out.is_dense = True
        out.data = filtered_raw.tobytes()
        self.pub.publish(out)

    def cloud_cb(self, msg: PointCloud2):
        n_pts = msg.width * msg.height
        if n_pts == 0 or msg.point_step == 0:
            return

        if self._cached_step != msg.point_step or self._cached_dtype is None:
            offsets = self._get_xyz_offsets(msg)
            if len(offsets) < 3:
                return
            self._cached_step = msg.point_step
            self._cached_dtype = np.dtype({
                'names': ['x', 'y', 'z'],
                'formats': ['<f4', '<f4', '<f4'],
                'offsets': [offsets['x'], offsets['y'], offsets['z']],
                'itemsize': msg.point_step,
            })

        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, msg.point_step)
        pts = np.frombuffer(msg.data, dtype=self._cached_dtype, count=n_pts)

        x = pts['x']
        y = pts['y']
        z = pts['z']

        # 1. Pre-filter: finite, below ceiling (max_z), outside robot body (min_range), within max_range
        r2 = x * x + y * y
        pre_mask = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
            (z >= self.min_z) & (z <= self.max_z) &
            (r2 >= self.min_r2) & (r2 <= self.max_r2)
        )
        if not np.any(pre_mask):
            self._publish_filtered(msg, raw[:0])
            return

        vx = x[pre_mask]
        vy = y[pre_mask]
        vz = z[pre_mask]
        vraw = raw[pre_mask]

        # 2. Bin into 2D (x, y) cells and compute per-cell delta_z = z_max - z_min
        half = self.max_range
        gd = self.grid_dim
        ix = np.clip(((vx + half) / self.cell_size).astype(np.int32), 0, gd - 1)
        iy = np.clip(((vy + half) / self.cell_size).astype(np.int32), 0, gd - 1)
        cell_id = iy * gd + ix

        n_cells = gd * gd
        z_min_grid = np.full(n_cells, np.inf, dtype=np.float32)
        z_max_grid = np.full(n_cells, -np.inf, dtype=np.float32)
        np.minimum.at(z_min_grid, cell_id, vz)
        np.maximum.at(z_max_grid, cell_id, vz)

        dz = z_max_grid[cell_id] - z_min_grid[cell_id]
        obs_mask = dz >= self.min_height_diff

        self._publish_filtered(msg, vraw[obs_mask])


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
