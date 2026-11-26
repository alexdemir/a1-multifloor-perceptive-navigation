#!/usr/bin/env python3
"""
Robot-Centric 2.5D Elevation Mapping Node for Perceptive Quadruped RL Locomotion.

Fuses Ouster OS1-32 3D LiDAR (/ouster/points) with DLIO 3D Odometry (/dlio/odom_node/odom)
into a rolling 5.0m x 5.0m (5 cm resolution, 100x100) 2.5D height grid in the odom frame.

Why odometry-fused rolling memory is essential:
  - The Ouster OS1-32 sits ~0.41m above ground with -22.5 deg min vertical angle, creating
    a ~1.0m radius blind cone directly underneath the robot.
  - Isaac Gym's 17x11 height_scan (x in [-0.8, +0.8]m, y in [-0.5, +0.5]m) lies entirely
    inside this blind cone.
  - By accumulating terrain heights at r in [0.42, 3.5]m into an integer-shifted 2.5D grid,
    stairs, curbs, and ramps scanned 1-2m ahead remain memorized right under the robot's feet.

Outputs:
  - /elevation_map/height_scan (std_msgs/Float32MultiArray, 50 Hz):
      187-dim legged_gym a1_rough observation: clip(base_z - 0.50 - terrain_z, -1, 1) * 5.0
  - /elevation_map/cloud (sensor_msgs/PointCloud2, 5 Hz):
      Live 2.5D elevation surface for RViz visualization.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray


class ElevationMapNode(Node):
    def __init__(self):
        super().__init__('elevation_map_node')

        self.declare_parameter('points_topic', '/ouster/points')
        self.declare_parameter('odom_topic', '/dlio/odom_node/odom')
        self.declare_parameter('grid_size_m', 5.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('lidar_z_offset', 0.09)
        self.declare_parameter('min_range', 0.42)
        self.declare_parameter('max_range', 3.50)
        self.declare_parameter('min_rel_z', -1.20)
        self.declare_parameter('max_rel_z', 0.45)
        self.declare_parameter('max_cell_dz', 0.24)
        self.declare_parameter('max_terrain_delta', 0.25)
        self.declare_parameter('nominal_base_height', 0.32)
        self.declare_parameter('ema_alpha', 0.45)

        points_topic = self.get_parameter('points_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.res = float(self.get_parameter('resolution').value)
        self.grid_n = int(round(float(self.get_parameter('grid_size_m').value) / self.res))
        self.lidar_z_offset = float(self.get_parameter('lidar_z_offset').value)
        self.min_r2 = float(self.get_parameter('min_range').value) ** 2
        self.max_r2 = float(self.get_parameter('max_range').value) ** 2
        self.min_rel_z = float(self.get_parameter('min_rel_z').value)
        self.max_rel_z = float(self.get_parameter('max_rel_z').value)
        self.max_cell_dz = float(self.get_parameter('max_cell_dz').value)
        self.max_terrain_delta = float(self.get_parameter('max_terrain_delta').value)
        self.nominal_h = float(self.get_parameter('nominal_base_height').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)

        # Rolling integer-indexed 2.5D grid in odom frame (NaN = unobserved)
        self.grid = np.full((self.grid_n, self.grid_n), np.nan, dtype=np.float32)
        self.origin_ix = None
        self.origin_iy = None

        # Latest odom + IMU state (position & yaw from odom, gravity-aligned roll/pitch from IMU)
        self.has_odom = False
        self.has_imu = False
        self.bx = 0.0
        self.by = 0.0
        self.bz = 0.0
        self.yaw = 0.0
        self.imu_roll = 0.0
        self.imu_pitch = 0.0
        self.rot_mat = np.eye(3, dtype=np.float32)

        # Isaac Gym legged_gym 17x11 height scan local offsets (row-major: x outer, y inner)
        xs = np.linspace(-0.8, 0.8, 17, dtype=np.float32)
        ys = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        self.scan_gx = gx.reshape(-1)  # (187,)
        self.scan_gy = gy.reshape(-1)  # (187,)

        self._cached_dtype = None
        self._cached_step = None
        self._last_cloud_pub_ns = 0
        self._last_scan_pub_ns = 0

        sub_cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
        )
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu/data', self.imu_cb, 20)
        self.sub_points = self.create_subscription(PointCloud2, points_topic, self.cloud_cb, sub_cloud_qos)
        self.pub_scan = self.create_publisher(Float32MultiArray, '/elevation_map/height_scan', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/elevation_map/cloud', 5)

        self.get_logger().info(
            f'2.5D ElevationMapNode ready ({self.grid_n}x{self.grid_n} @ {self.res:.2f}m, '
            f'17x11 height_scan -> /elevation_map/height_scan)'
        )

    def _shift_grid_to_robot(self):
        half = self.grid_n // 2
        target_ix = int(math.floor(self.bx / self.res)) - half
        target_iy = int(math.floor(self.by / self.res)) - half

        if self.origin_ix is None:
            self.origin_ix = target_ix
            self.origin_iy = target_iy
            return

        dx = target_ix - self.origin_ix
        dy = target_iy - self.origin_iy
        if dx == 0 and dy == 0:
            return

        if abs(dx) >= self.grid_n or abs(dy) >= self.grid_n:
            self.grid.fill(np.nan)
        else:
            if dx != 0:
                self.grid = np.roll(self.grid, -dx, axis=1)
                if dx > 0:
                    self.grid[:, -dx:] = np.nan
                else:
                    self.grid[:, :-dx] = np.nan
            if dy != 0:
                self.grid = np.roll(self.grid, -dy, axis=0)
                if dy > 0:
                    self.grid[-dy:, :] = np.nan
                else:
                    self.grid[:-dy, :] = np.nan

        self.origin_ix = target_ix
        self.origin_iy = target_iy

    def _update_rot_mat(self):
        cr, sr = math.cos(self.imu_roll), math.sin(self.imu_roll)
        cp, sp = math.cos(self.imu_pitch), math.sin(self.imu_pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.rot_mat = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ], dtype=np.float32)

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        self.imu_roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sinp = 2.0 * (qw * qy - qz * qx)
        self.imu_pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        self.has_imu = True
        self._update_rot_mat()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.bx = float(p.x)
        self.by = float(p.y)
        self.bz = float(p.z)

        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        self.yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        if not self.has_imu:
            self.imu_roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
            sinp = 2.0 * (qw * qy - qz * qx)
            self.imu_pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        self._update_rot_mat()
        self.has_odom = True
        self._shift_grid_to_robot()

        # Publish 187-dim height_scan at ~50 Hz synchronized with sim odom stamps
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds
        if (stamp_ns - self._last_scan_pub_ns) >= 18_000_000:
            self._last_scan_pub_ns = stamp_ns
            self._publish_height_scan()

    def _get_xyz_offsets(self, msg: PointCloud2):
        offsets = {}
        for f in msg.fields:
            if f.name in ('x', 'y', 'z'):
                offsets[f.name] = f.offset
        return offsets

    def cloud_cb(self, msg: PointCloud2):
        if not self.has_odom or self.origin_ix is None:
            return
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

        pts = np.frombuffer(msg.data, dtype=self._cached_dtype, count=n_pts)
        lx = pts['x']
        ly = pts['y']
        lz = pts['z']

        # 1. Pre-filter in lidar_link frame (exclude self-hits and distant walls)
        r2 = lx * lx + ly * ly
        mask = np.isfinite(lx) & np.isfinite(ly) & np.isfinite(lz) & (r2 >= self.min_r2) & (r2 <= self.max_r2)
        if not np.any(mask):
            return

        # 2. Transform lidar_link -> base -> odom
        p_base = np.empty((int(np.count_nonzero(mask)), 3), dtype=np.float32)
        p_base[:, 0] = lx[mask]
        p_base[:, 1] = ly[mask]
        p_base[:, 2] = lz[mask] + self.lidar_z_offset

        p_odom = p_base @ self.rot_mat.T
        wx = p_odom[:, 0] + self.bx
        wy = p_odom[:, 1] + self.by
        wz = p_odom[:, 2] + self.bz

        # 3. Height filter in odom frame (keep floors/stairs/ramps, reject ceilings/high wall tops)
        z_mask = (wz >= (self.bz + self.min_rel_z)) & (wz <= (self.bz + self.max_rel_z))
        if not np.any(z_mask):
            return
        wx = wx[z_mask]
        wy = wy[z_mask]
        wz = wz[z_mask]

        # 4. Bin into 2.5D rolling grid; reject vertical walls (cell_max - cell_min > max_cell_dz)
        #    while preserving flat ground, 14-deg ramps, and <=18cm stair risers.
        ix = np.floor(wx / self.res).astype(np.int32) - self.origin_ix
        iy = np.floor(wy / self.res).astype(np.int32) - self.origin_iy
        in_bounds = (ix >= 0) & (ix < self.grid_n) & (iy >= 0) & (iy < self.grid_n)
        if not np.any(in_bounds):
            return

        ix = ix[in_bounds]
        iy = iy[in_bounds]
        wz = wz[in_bounds]

        cell_min = np.full((self.grid_n, self.grid_n), np.inf, dtype=np.float32)
        cell_max = np.full((self.grid_n, self.grid_n), -np.inf, dtype=np.float32)
        cell_sum = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        cell_cnt = np.zeros((self.grid_n, self.grid_n), dtype=np.int32)
        np.minimum.at(cell_min, (iy, ix), wz)
        np.maximum.at(cell_max, (iy, ix), wz)
        np.add.at(cell_sum, (iy, ix), wz)
        np.add.at(cell_cnt, (iy, ix), 1)

        has_pts = cell_cnt > 0
        cell_dz = np.where(has_pts, cell_max - cell_min, 0.0)
        cell_mean = np.where(has_pts, cell_sum / np.maximum(cell_cnt, 1), 0.0)
        # Exclude vertical walls/actors (cell_dz > max_cell_dz) while keeping uphill ramps & stairs ahead
        is_wall = has_pts & (cell_dz > self.max_cell_dz)
        self.grid[is_wall] = np.nan
        updated = has_pts & (~is_wall)

        # 5. EMA fusion of unbiased cell mean into persistent rolling elevation grid
        existing = updated & np.isfinite(self.grid)
        fresh = updated & (~np.isfinite(self.grid))
        self.grid[fresh] = cell_mean[fresh]
        self.grid[existing] = (1.0 - self.ema_alpha) * self.grid[existing] + self.ema_alpha * cell_mean[existing]

        # 6. Publish RViz visualization cloud at 5 Hz
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds
        if (stamp_ns - self._last_cloud_pub_ns) >= 200_000_000:
            self._last_cloud_pub_ns = stamp_ns
            self._publish_elevation_cloud(msg.header.stamp)

    def _publish_height_scan(self):
        if self.origin_ix is None:
            return

        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        # Rotate 17x11 scan grid into horizontal base-yaw frame (exact legged_gym convention)
        wx = self.bx + cy * self.scan_gx - sy * self.scan_gy
        wy = self.by + sy * self.scan_gx + cy * self.scan_gy

        ix = np.floor(wx / self.res).astype(np.int32) - self.origin_ix
        iy = np.floor(wy / self.res).astype(np.int32) - self.origin_iy
        in_bounds = (ix >= 0) & (ix < self.grid_n) & (iy >= 0) & (iy < self.grid_n)

        terrain_z = np.full(187, np.nan, dtype=np.float32)
        terrain_z[in_bounds] = self.grid[iy[in_bounds], ix[in_bounds]]

        # 1-pass 3x3 neighbor fill for tiny holes inside the LiDAR blind cone
        missing = np.where(~np.isfinite(terrain_z) & in_bounds)[0]
        if missing.size > 0:
            for idx in missing:
                cx_i = int(ix[idx])
                cy_i = int(iy[idx])
                y0 = max(0, cy_i - 1)
                y1 = min(self.grid_n, cy_i + 2)
                x0 = max(0, cx_i - 1)
                x1 = min(self.grid_n, cx_i + 2)
                patch = self.grid[y0:y1, x0:x1]
                valid_vals = patch[np.isfinite(patch)]
                if valid_vals.size > 0:
                    terrain_z[idx] = float(np.median(valid_vals))

        # Fallback for unobserved/wall cells -> flat ground under robot, and clamp to in-distribution range
        ground_ref = self.bz - self.nominal_h
        still_nan = ~np.isfinite(terrain_z)
        if np.any(still_nan):
            terrain_z[still_nan] = ground_ref
        terrain_z = np.clip(terrain_z, ground_ref - self.max_terrain_delta, ground_ref + self.max_terrain_delta)

        # Exact legged_gym a1_rough observation scaling: clip(base_z - 0.5 - terrain_z, -1, 1) * 5.0
        obs_h = np.clip(self.bz - 0.50 - terrain_z, -1.0, 1.0).astype(np.float32) * 5.0

        msg = Float32MultiArray()
        msg.data = obs_h.tolist()
        self.pub_scan.publish(msg)

    def _publish_elevation_cloud(self, stamp):
        if self.pub_cloud.get_subscription_count() == 0:
            return
        valid_iy, valid_ix = np.where(np.isfinite(self.grid))
        if valid_ix.size == 0:
            return

        pts = np.empty((valid_ix.size, 3), dtype=np.float32)
        pts[:, 0] = (valid_ix + self.origin_ix + 0.5) * self.res
        pts[:, 1] = (valid_iy + self.origin_iy + 0.5) * self.res
        pts[:, 2] = self.grid[valid_iy, valid_ix]

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = 'odom'
        cloud.height = 1
        cloud.width = int(valid_ix.size)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.width * 12
        cloud.is_dense = True
        cloud.data = pts.tobytes()
        self.pub_cloud.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = ElevationMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
