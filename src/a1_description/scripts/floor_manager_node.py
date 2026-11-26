#!/usr/bin/env python3
"""
Topological N-Floor Manager & Hierarchical Cross-Floor Navigator for Nav2.

Reads building_topology.yaml (G = (V, E)) defining:
  - Floors (V): name, numeric id, map yaml, z_center
  - Connectors (E): ramps linking lower_floor <-> upper_floor with bottom_xy, top_xy,
    handoff_radius, yaw_up, yaw_down, and traversal cost.

Capabilities:
  1. Generalized N-floor elevation tracking from DLIO 3D odometry (/dlio/odom_node/odom)
     with hysteresis around inter-floor midpoints.
  2. Publishes /current_floor (Int32), /current_floor_name (String), and
     /floor_manager/mission_status (String: IDLE/RUNNING/SUCCEEDED/FAILED).
  3. Dijkstra shortest-path graph planning across any number of floors (e.g. floor1 -> floor3):
     automatically chains ramp handoffs, dynamic /map_server/load_map calls, and costmap clears.
"""

import heapq
import math
import os
import threading
import time
import yaml
import numpy as np
from PIL import Image

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap, LoadMap
from action_msgs.msg import GoalStatus


class FloorManagerNode(Node):
    def __init__(self):
        super().__init__('floor_manager_node')
        self.cb_group = ReentrantCallbackGroup()

        pkg_dir = get_package_share_directory('a1_description')
        default_topology = os.path.join(pkg_dir, 'config', 'building_topology.yaml')
        maps_dir = os.path.join(pkg_dir, 'maps')

        self.declare_parameter('topology_file', default_topology)
        self.declare_parameter('odom_topic', '/dlio/odom_node/odom')
        self.declare_parameter('z_hysteresis', 0.30)

        topology_file = self.get_parameter('topology_file').value
        self.z_hysteresis = float(self.get_parameter('z_hysteresis').value)

        with open(topology_file, 'r') as f:
            topo = yaml.safe_load(f)

        # Parse floors: name -> {id, map_path, z_center}
        self.floors = {}
        self.id_to_floor = {}
        for fname, fdata in topo['floors'].items():
            mpath = fdata['map']
            if not os.path.isabs(mpath):
                mpath = os.path.join(maps_dir, mpath)
            self.floors[fname] = {
                'id': int(fdata['id']),
                'map_path': mpath,
                'z_center': float(fdata['z_center']),
            }
            self.id_to_floor[int(fdata['id'])] = fname

        # Sort floors by z_center for N-floor hysteresis elevation tracking
        self.sorted_floors = sorted(self.floors.keys(), key=lambda fn: self.floors[fn]['z_center'])

        # Build bidirectional topological adjacency list from connectors
        self.connectors = topo.get('connectors', [])
        self.adj = {fn: [] for fn in self.floors}
        for conn in self.connectors:
            lo = conn['lower_floor']
            hi = conn['upper_floor']
            cost = float(conn.get('cost', 10.0))
            self.adj[lo].append((hi, conn, 'up', cost))
            self.adj[hi].append((lo, conn, 'down', cost))

        # Preload each floor's 2D occupancy grid for goal validation
        self.floor_grids = {}
        for fname, fdata in self.floors.items():
            try:
                with open(fdata['map_path'], 'r') as mf:
                    mcfg = yaml.safe_load(mf)
                img_path = mcfg['image']
                if not os.path.isabs(img_path):
                    img_path = os.path.join(os.path.dirname(fdata['map_path']), img_path)
                grid = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
                self.floor_grids[fname] = {
                    'grid': grid,
                    'res': float(mcfg['resolution']),
                    'ox': float(mcfg['origin'][0]),
                    'oy': float(mcfg['origin'][1]),
                }
            except Exception as exc:
                self.get_logger().warn(f'Could not preload grid for {fname}: {exc}')

        self.current_floor = self.sorted_floors[0]
        self.loaded_map_floor = self.sorted_floors[0]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
        self.mission_active = False
        self._lock = threading.Lock()
        self._switch_in_progress = False

        # Subscribers & Publishers
        odom_topic = self.get_parameter('odom_topic').value
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 10, callback_group=self.cb_group
        )
        self.sub_goal = self.create_subscription(
            PoseStamped, '/multi_floor_goal', self.multi_floor_goal_cb, 10, callback_group=self.cb_group
        )
        self.pub_floor = self.create_publisher(Int32, '/current_floor', 10)
        self.pub_floor_name = self.create_publisher(String, '/current_floor_name', 10)
        self.pub_status = self.create_publisher(String, '/floor_manager/mission_status', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Service & Action Clients
        self.cli_load_map = self.create_client(
            LoadMap, '/map_server/load_map', callback_group=self.cb_group
        )
        self.cli_clear_global = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap', callback_group=self.cb_group
        )
        self.cli_clear_local = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap', callback_group=self.cb_group
        )
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb_group
        )

        self.timer = self.create_timer(0.5, self.publish_floor_state, callback_group=self.cb_group)
        self.get_logger().info(
            f'Topological FloorManager ready ({len(self.floors)} floors: {self.sorted_floors}, '
            f'{len(self.connectors)} connectors loaded from {topology_file})'
        )

    def publish_floor_state(self):
        msg_id = Int32()
        msg_id.data = self.floors[self.current_floor]['id']
        self.pub_floor.publish(msg_id)

        msg_name = String()
        msg_name.data = self.current_floor
        self.pub_floor_name.publish(msg_name)

    def _publish_mission_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)

    def odom_cb(self, msg: Odometry):
        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)
        self.current_z = float(msg.pose.pose.position.z)
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        # N-floor elevation tracking with hysteresis around inter-floor midpoints
        idx = self.sorted_floors.index(self.current_floor)
        new_floor = self.current_floor

        if idx + 1 < len(self.sorted_floors):
            upper = self.sorted_floors[idx + 1]
            mid_up = 0.5 * (self.floors[self.current_floor]['z_center'] + self.floors[upper]['z_center'])
            if self.current_z > mid_up + self.z_hysteresis:
                new_floor = upper

        if idx - 1 >= 0:
            lower = self.sorted_floors[idx - 1]
            mid_down = 0.5 * (self.floors[self.current_floor]['z_center'] + self.floors[lower]['z_center'])
            if self.current_z < mid_down - self.z_hysteresis:
                new_floor = lower

        if new_floor != self.current_floor:
            self.current_floor = new_floor
            self.get_logger().info(
                f'Physical floor changed -> {self.current_floor} (z={self.current_z:.2f}m)'
            )

        # Passive mode (teleop or RViz 2D Goal Pose): auto-sync map with physical floor
        if not self.mission_active and self.loaded_map_floor != self.current_floor and not self._switch_in_progress:
            threading.Thread(target=self.switch_map_sync, args=(self.current_floor,), daemon=True).start()

    def _call_service_sync(self, client, request, timeout_sec=5.0):
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().warn(f'Service {client.srv_name} not available.')
            return None
        future = client.call_async(request)
        t0 = time.monotonic()
        while rclpy.ok() and not future.done():
            if time.monotonic() - t0 > timeout_sec:
                self.get_logger().warn(f'Service {client.srv_name} timed out.')
                return None
            time.sleep(0.05)
        return future.result()

    def switch_map_sync(self, target_floor: str) -> bool:
        with self._lock:
            if self.loaded_map_floor == target_floor:
                return True
            self._switch_in_progress = True

        try:
            map_url = self.floors[target_floor]['map_path']
            self.get_logger().info(f'Switching Nav2 map -> {target_floor} ({map_url})')
            req = LoadMap.Request()
            req.map_url = map_url
            res = self._call_service_sync(self.cli_load_map, req, timeout_sec=6.0)
            if res is None or res.result != LoadMap.Response.RESULT_SUCCESS:
                self.get_logger().error(f'Failed to load {target_floor} map!')
                return False

            self._call_service_sync(self.cli_clear_global, ClearEntireCostmap.Request(), timeout_sec=3.0)
            self._call_service_sync(self.cli_clear_local, ClearEntireCostmap.Request(), timeout_sec=3.0)
            time.sleep(0.4)

            with self._lock:
                self.loaded_map_floor = target_floor
            self.get_logger().info(f'{target_floor} map active and costmaps cleared.')
            return True
        finally:
            with self._lock:
                self._switch_in_progress = False

    def plan_floor_path(self, start_floor: str, target_floor: str):
        """Dijkstra shortest path on the building topology graph."""
        if start_floor == target_floor:
            return []

        pq = [(0.0, start_floor, [])]
        visited = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == target_floor:
                return path
            if node in visited:
                continue
            visited.add(node)
            for nxt, conn, direction, edge_cost in self.adj.get(node, []):
                if nxt not in visited:
                    heapq.heappush(pq, (cost + edge_cost, nxt, path + [(nxt, conn, direction)]))
        return None

    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    def _send_nav_goal_sync(self, pose: PoseStamped, label: str, stop_within_radius: float = None) -> bool:
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 navigate_to_pose action server not available.')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(
            f'[{label}] Sending Nav2 goal: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})'
        )
        send_future = self.nav_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_future.done():
            time.sleep(0.05)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'[{label}] Goal rejected by Nav2.')
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if stop_within_radius is not None:
                dist = math.hypot(
                    self.current_x - pose.pose.position.x,
                    self.current_y - pose.pose.position.y,
                )
                if dist <= stop_within_radius:
                    self.get_logger().info(
                        f'[{label}] Entered handoff zone (dist={dist:.2f}m <= {stop_within_radius:.2f}m). Transitioning...'
                    )
                    cancel_fut = goal_handle.cancel_goal_async()
                    t_c = time.monotonic()
                    while rclpy.ok() and not cancel_fut.done() and (time.monotonic() - t_c) < 2.0:
                        time.sleep(0.05)
                    time.sleep(0.3)
                    return True
            time.sleep(0.1)

        result = result_future.result()
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'[{label}] Goal reached successfully!')
            return True

        status = result.status if result is not None else -1
        self.get_logger().error(f'[{label}] Navigation failed with status={status}')
        return False

    def _parse_target_floor(self, frame_id: str, z_val: float = 0.0) -> str:
        fid = frame_id.strip().lower()
        if fid in self.floors:
            return fid
        if fid.isdigit() and int(fid) in self.id_to_floor:
            return self.id_to_floor[int(fid)]
        if fid.startswith('f') and fid[1:].isdigit() and int(fid[1:]) in self.id_to_floor:
            return self.id_to_floor[int(fid[1:])]
        # Fallback: match closest floor by pose.position.z (e.g. frame_id == 'map')
        return min(
            self.sorted_floors,
            key=lambda fn: abs(self.floors[fn]['z_center'] - float(z_val))
        )

    def _validate_goal(self, target_floor: str, x: float, y: float):
        """Reject goals on ramps, inside walls, in unknown space, or outside the floor map."""
        for conn in self.connectors:
            if conn.get('upper_floor') != target_floor:
                continue
            bx, by = float(conn['bottom_xy'][0]), float(conn['bottom_xy'][1])
            tx, ty = float(conn['top_xy'][0]), float(conn['top_xy'][1])
            dx, dy = tx - bx, ty - by
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux
            s = (x - bx) * ux + (y - by) * uy
            d = (x - bx) * nx + (y - by) * ny
            if 0.75 <= s <= (length - 0.55) and abs(d) <= 1.15:
                return False, f"Target ({x:.2f}, {y:.2f}) is on {conn['name']} (ramps are forbidden)"

        fg = self.floor_grids.get(target_floor)
        if fg is not None:
            grid = fg['grid']
            res, ox, oy = fg['res'], fg['ox'], fg['oy']
            h, w = grid.shape
            col = int((x - ox) / res)
            row = h - 1 - int((y - oy) / res)
            if not (0 <= col < w and 0 <= row < h):
                return False, f'Target ({x:.2f}, {y:.2f}) is outside {target_floor} map bounds'
            val = int(grid[row, col])
            if val < 100:
                return False, f'Target ({x:.2f}, {y:.2f}) is inside a wall/obstacle on {target_floor}'
            if val < 230:
                return False, f'Target ({x:.2f}, {y:.2f}) is not walkable on {target_floor}'
            r_px = max(1, int(round(0.25 / res)))
            patch = grid[max(0, row - r_px):min(h, row + r_px + 1), max(0, col - r_px):min(w, col + r_px + 1)]
            if np.any(patch < 100):
                return False, f'Target ({x:.2f}, {y:.2f}) is too close to a wall (<0.25m)'

        return True, 'OK'

    def multi_floor_goal_cb(self, msg: PoseStamped):
        target_floor = self._parse_target_floor(msg.header.frame_id, msg.pose.position.z)
        ok, reason = self._validate_goal(target_floor, msg.pose.position.x, msg.pose.position.y)
        if not ok:
            self.get_logger().warn(f'Goal rejected: {reason}')
            self._publish_mission_status(f'REJECTED: {reason}')
            return
        if self.mission_active:
            self.get_logger().warn('A multi-floor mission is already active; ignoring new goal.')
            return

        threading.Thread(
            target=self._execute_mission, args=(target_floor, msg), daemon=True
        ).start()

    def _align_yaw_sync(self, target_yaw: float, tol: float = 0.025, timeout_sec: float = 18.0):
        """Rotate in place on flat ground so the robot enters an uphill ramp straight."""
        t0 = time.monotonic()
        aligned = False
        while rclpy.ok() and (time.monotonic() - t0) < timeout_sec:
            err = math.atan2(
                math.sin(target_yaw - self.current_yaw),
                math.cos(target_yaw - self.current_yaw),
            )
            if abs(err) <= tol:
                aligned = True
                break
            cmd = Twist()
            wz = max(-0.55, min(0.55, 1.3 * err))
            if abs(wz) < 0.18:
                wz = math.copysign(0.18, err)
            cmd.angular.z = float(wz)
            self.pub_cmd_vel.publish(cmd)
            time.sleep(0.05)

        self.pub_cmd_vel.publish(Twist())
        time.sleep(0.25)
        self.get_logger().info(
            f'Pre-ramp yaw alignment {"OK" if aligned else "timeout"} '
            f'(yaw={math.degrees(self.current_yaw):.1f} deg, target={math.degrees(target_yaw):.1f} deg)'
        )

    def _center_on_entry_sync(self, entry_xy, target_yaw: float, tol_xy: float = 0.15, timeout_sec: float = 10.0):
        """Translate on flat ground to the exact ramp centerline before climbing."""
        ex, ey = float(entry_xy[0]), float(entry_xy[1])
        tx, ty = math.cos(target_yaw), math.sin(target_yaw)
        nx, ny = -ty, tx
        t0 = time.monotonic()
        centered = False
        while rclpy.ok() and (time.monotonic() - t0) < timeout_sec:
            dx = ex - self.current_x
            dy = ey - self.current_y
            dist = math.hypot(dx, dy)
            yaw_err = math.atan2(
                math.sin(target_yaw - self.current_yaw),
                math.cos(target_yaw - self.current_yaw),
            )
            if dist <= tol_xy and abs(yaw_err) <= 0.03:
                centered = True
                break

            err_fwd = dx * tx + dy * ty
            err_lat = dx * nx + dy * ny
            cmd = Twist()
            cmd.linear.x = float(max(-0.28, min(0.28, 1.4 * err_fwd)))
            cmd.linear.y = float(max(-0.24, min(0.24, 1.4 * err_lat)))
            wz = max(-0.45, min(0.45, 1.8 * yaw_err))
            if abs(yaw_err) > 0.025 and abs(wz) < 0.18:
                wz = math.copysign(0.18, yaw_err)
            cmd.angular.z = float(wz)
            self.pub_cmd_vel.publish(cmd)
            time.sleep(0.05)

        self.pub_cmd_vel.publish(Twist())
        time.sleep(0.25)
        self.get_logger().info(
            f'Pre-ramp centering {"OK" if centered else "timeout"} '
            f'at ({self.current_x:.2f}, {self.current_y:.2f})'
        )

    def _traverse_ramp_sync(self, start_xy, end_xy, target_yaw: float, is_uphill: bool, timeout_sec: float = 150.0) -> bool:
        """
        Deterministic ramp/stair corridor traversal:
        Holds straight via 50 Hz IMU yaw lock inside the centerline deadzone and gently steers back
        if drifting toward guard walls, avoiding Nav2 Pure Pursuit oscillations on narrow connectors.
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        ex, ey = float(end_xy[0]), float(end_xy[1])
        dx, dy = ex - sx, ey - sy
        ramp_len = math.hypot(dx, dy)
        if ramp_len < 0.1:
            return True

        tx, ty = dx / ramp_len, dy / ramp_len
        nx, ny = -ty, tx  # unit left-normal of the ramp centerline
        fwd_speed = 0.65 if is_uphill else 0.42

        self.get_logger().info(
            f'Ramp Mode START ({"UP" if is_uphill else "DOWN"}): '
            f'({sx:.2f},{sy:.2f}) -> ({ex:.2f},{ey:.2f}), yaw={math.degrees(target_yaw):.1f} deg'
        )

        t0 = time.monotonic()
        reached = False
        while rclpy.ok() and (time.monotonic() - t0) < timeout_sec:
            rx = self.current_x - sx
            ry = self.current_y - sy
            progress = rx * tx + ry * ty
            cte = rx * nx + ry * ny  # positive = drifted left of centerline

            if progress >= (ramp_len - 0.25):
                reached = True
                break

            yaw_err = math.atan2(
                math.sin(target_yaw - self.current_yaw),
                math.cos(target_yaw - self.current_yaw),
            )

            cmd = Twist()
            cmd.linear.x = float(fwd_speed)
            cmd.linear.y = 0.0
            if not is_uphill:
                # Descent centerline tracking: keep body nearly parallel (<= 8 deg offset)
                if abs(yaw_err) <= 0.05 and abs(cte) <= 0.12:
                    cmd.angular.z = 0.0
                else:
                    desired_yaw = target_yaw - max(-0.20, min(0.20, 0.50 * cte))
                    steer_err = math.atan2(
                        math.sin(desired_yaw - self.current_yaw),
                        math.cos(desired_yaw - self.current_yaw),
                    )
                    if abs(steer_err) > 0.20:
                        cmd.linear.x = 0.15
                    cmd.angular.z = float(max(-0.40, min(0.40, 2.20 * steer_err)))
            elif abs(yaw_err) <= 0.04 and abs(cte) <= 0.06:
                # Straight & centered: let rl_locomotion_controller's 50 Hz IMU yaw lock hold straight
                cmd.angular.z = 0.0
            else:
                # Keep body straight on ramps/stairs and strafe gently with linear.y (no yaw twisting)
                cmd.linear.y = float(max(-0.18, min(0.18, -0.55 * cte)))
                if abs(yaw_err) > 0.25:
                    cmd.linear.x = 0.38
                cmd.angular.z = float(max(-0.35, min(0.35, 1.50 * yaw_err)))
            self.pub_cmd_vel.publish(cmd)
            time.sleep(0.05)

        self.pub_cmd_vel.publish(Twist())
        time.sleep(0.30)
        self.get_logger().info(
            f'Ramp Mode {"COMPLETED" if reached else "TIMEOUT"} at '
            f'({self.current_x:.2f}, {self.current_y:.2f}, z={self.current_z:.2f})'
        )
        return reached

    def _execute_mission(self, target_floor: str, final_goal: PoseStamped):
        self.mission_active = True
        self._publish_mission_status('RUNNING')
        mission_ok = False
        try:
            start_floor = self.current_floor
            floor_path = self.plan_floor_path(start_floor, target_floor)
            if floor_path is None:
                self.get_logger().error(f'No topological route from {start_floor} to {target_floor}!')
                return

            route_str = ' -> '.join([start_floor] + [f"{c['name']} -> {nf}" for nf, c, _ in floor_path])
            self.get_logger().info(
                f'Hierarchical mission started: {route_str} | '
                f'target=({final_goal.pose.position.x:.2f}, {final_goal.pose.position.y:.2f})'
            )

            if not self.switch_map_sync(start_floor):
                return

            # Traverse each topological connector in the Dijkstra route
            for step_idx, (next_floor, conn, direction) in enumerate(floor_path, start=1):
                radius = float(conn.get('handoff_radius', 0.55))
                if direction == 'up':
                    entry_xy = conn['bottom_xy']
                    exit_xy = conn['top_xy']
                    yaw = float(conn['yaw_up'])
                else:
                    entry_xy = conn['top_xy']
                    exit_xy = conn['bottom_xy']
                    yaw = float(conn['yaw_down'])

                handoff_pose = self._make_pose(entry_xy[0], entry_xy[1], yaw)
                ok = self._send_nav_goal_sync(
                    handoff_pose,
                    f"Step {step_idx} ({conn['name']} {direction} -> entry)",
                    stop_within_radius=radius,
                )
                if not ok:
                    self.get_logger().error(f"Aborting mission: failed reaching {conn['name']} entry.")
                    return

                self._align_yaw_sync(yaw)
                self._center_on_entry_sync(entry_xy, yaw)

                if direction == 'up':
                    if not self.switch_map_sync(next_floor):
                        self.get_logger().error(f'Aborting mission: failed switching to {next_floor} map.')
                        return
                    if not self._traverse_ramp_sync(entry_xy, exit_xy, yaw, is_uphill=True):
                        self.get_logger().error(f"Aborting mission: failed climbing {conn['name']}.")
                        return
                else:
                    if not self._traverse_ramp_sync(entry_xy, exit_xy, yaw, is_uphill=False):
                        self.get_logger().error(f"Aborting mission: failed descending {conn['name']}.")
                        return
                    if not self.switch_map_sync(next_floor):
                        self.get_logger().error(f'Aborting mission: failed switching to {next_floor} map.')
                        return

            # Final leg on target_floor's map (automatically climbs last ramp if ascending)
            ok_final = self._send_nav_goal_sync(
                final_goal, f'Final Leg ({target_floor})'
            )
            if ok_final:
                mission_ok = True
                self.get_logger().info(f'Multi-floor mission to {target_floor} COMPLETED!')
        finally:
            self._publish_mission_status('SUCCEEDED' if mission_ok else 'FAILED')
            self.mission_active = False


def main(args=None):
    rclpy.init(args=args)
    node = FloorManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
