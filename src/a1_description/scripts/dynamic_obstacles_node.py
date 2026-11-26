#!/usr/bin/env python3
"""
Dynamic Moving Obstacles Controller for 3-Story Gazebo Building (On-Demand Spawn + Social Yielding).

Dynamically spawns 6 human-sized actors (2 per floor: Floor 1, Floor 2, Floor 3) via
/spawn_entity so the .world file stays 100% clean during mapping. Moves them smoothly
along patrol corridors via /gazebo/set_entity_state with real collision geometry so
Ouster 3D LiDAR and Nav2 global/local costmaps detect and replan around them.

Each actor politely yields (pauses) when the robot approaches within 1.30m on the same
floor and resumes walking once the robot clears 1.55m.
"""

import math
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState, SpawnEntity


def build_actor_sdf(name: str, rgb: str) -> str:
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <link name="body">
      <gravity>false</gravity>
      <kinematic>true</kinematic>
      <collision name="torso_col">
        <pose>0 0 0.65 0 0 0</pose>
        <geometry><cylinder><radius>0.22</radius><length>1.15</length></cylinder></geometry>
      </collision>
      <visual name="torso_vis">
        <pose>0 0 0.65 0 0 0</pose>
        <geometry><cylinder><radius>0.22</radius><length>1.15</length></cylinder></geometry>
        <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>
      </visual>
      <collision name="head_col">
        <pose>0 0 1.38 0 0 0</pose>
        <geometry><sphere><radius>0.14</radius></sphere></geometry>
      </collision>
      <visual name="head_vis">
        <pose>0 0 1.38 0 0 0</pose>
        <geometry><sphere><radius>0.14</radius></sphere></geometry>
        <material><ambient>0.95 0.8 0.65 1</ambient><diffuse>0.95 0.8 0.65 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


class DynamicObstaclesNode(Node):
    def __init__(self):
        super().__init__('dynamic_obstacles_node')

        self.declare_parameter('speed', 0.35)
        self.declare_parameter('yield_distance', 1.45)
        self.declare_parameter('resume_distance', 1.70)
        self.speed = float(self.get_parameter('speed').value)
        self.yield_dist = float(self.get_parameter('yield_distance').value)
        self.resume_dist = float(self.get_parameter('resume_distance').value)

        self.cli_spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self.cli_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.sub_models = self.create_subscription(
            ModelStates, '/gazebo/model_states', self.model_states_cb, 10
        )

        self.robot_pos = None
        self.existing_models = set()
        self.spawned = set()
        self.dt = 0.1  # 10 Hz smooth motion

        # (name, x_start, y_start, x_end, y_end, z, rgb)
        self.actors = [
            # Floor 1 (z = 0.00m, red/orange)
            ('actor_f1_a',  2.2, -1.5,  2.2,  3.0, 0.00, '0.90 0.25 0.15'),
            ('actor_f1_b', -4.5,  0.5, -4.5,  4.5, 0.00, '0.90 0.45 0.10'),
            # Floor 2 (z = 1.50m, blue/cyan)
            ('actor_f2_a',  2.2, -1.8,  2.2,  1.8, 1.50, '0.15 0.55 0.90'),
            ('actor_f2_b',  5.2, -2.8,  5.2,  2.5, 1.50, '0.10 0.75 0.75'),
            # Floor 3 (z = 3.00m, purple/magenta)
            ('actor_f3_a', -2.0, -3.2, -2.0,  1.0, 3.00, '0.70 0.20 0.85'),
            ('actor_f3_b', -5.3, -3.5, -5.3,  1.2, 3.00, '0.85 0.20 0.55'),
        ]
        self.actor_t = {a[0]: 0.0 for a in self.actors}
        self.actor_yielding = {a[0]: False for a in self.actors}

        self.spawn_timer = self.create_timer(1.0, self.try_spawn_actors)
        self.timer = self.create_timer(self.dt, self.step_cb)
        self.get_logger().info(
            f'DynamicObstaclesNode ready ({len(self.actors)} on-demand actors across 3 floors, '
            f'speed={self.speed:.2f} m/s, yield<{self.yield_dist:.2f}m)'
        )

    def model_states_cb(self, msg: ModelStates):
        self.existing_models = set(msg.name)
        if 'a1_quadruped' in msg.name:
            idx = msg.name.index('a1_quadruped')
            p = msg.pose[idx].position
            self.robot_pos = (float(p.x), float(p.y), float(p.z))

    def try_spawn_actors(self):
        # Wait until Gazebo factory & robot are up so gzclient renders cleanly
        if 'a1_quadruped' not in self.existing_models or not self.cli_spawn.service_is_ready():
            return

        for name, x0, y0, _, _, z, rgb in self.actors:
            if name in self.spawned or name in self.existing_models:
                self.spawned.add(name)
                continue
            req = SpawnEntity.Request()
            req.name = name
            req.xml = build_actor_sdf(name, rgb)
            req.initial_pose.position.x = float(x0)
            req.initial_pose.position.y = float(y0)
            req.initial_pose.position.z = float(z)
            req.initial_pose.orientation.w = 1.0
            self.cli_spawn.call_async(req)
            self.spawned.add(name)
            self.get_logger().info(f'Spawned dynamic actor: {name} at ({x0:.1f}, {y0:.1f}, z={z:.2f})')

        if len(self.spawned) == len(self.actors):
            self.spawn_timer.cancel()

    def step_cb(self):
        if not self.cli_state.service_is_ready():
            return

        for name, x0, y0, x1, y1, z, _ in self.actors:
            if name not in self.existing_models:
                continue

            dist_seg = math.hypot(x1 - x0, y1 - y0)
            period = (2.0 * dist_seg) / max(self.speed, 0.05)

            alpha = 0.5 * (1.0 - math.cos(2.0 * math.pi * self.actor_t[name] / period))
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)

            if self.robot_pos is not None:
                rx, ry, rz = self.robot_pos
                same_floor = abs(rz - (z + 0.30)) < 0.80
                d_robot = math.hypot(rx - x, ry - y) if same_floor else 999.0
                if not self.actor_yielding[name] and d_robot < self.yield_dist:
                    self.actor_yielding[name] = True
                elif self.actor_yielding[name] and d_robot > self.resume_dist:
                    self.actor_yielding[name] = False

            if not self.actor_yielding[name]:
                self.actor_t[name] += self.dt
                alpha = 0.5 * (1.0 - math.cos(2.0 * math.pi * self.actor_t[name] / period))
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)

            req = SetEntityState.Request()
            req.state.name = name
            req.state.reference_frame = 'world'
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.position.z = float(z)
            req.state.pose.orientation.w = 1.0
            self.cli_state.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstaclesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
