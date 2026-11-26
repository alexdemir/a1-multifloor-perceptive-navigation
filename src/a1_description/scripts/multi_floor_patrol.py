#!/usr/bin/env python3
"""
Autonomous 3-Story Multi-Floor Patrol Mission Runner.

Demonstrates hierarchical Dijkstra routing across all 3 floors + 2 ramps while
avoiding 6 moving dynamic actors:
  1. Floor 1 East Room     ( 3.2,  1.8)
  2. Floor 2 East Corridor ( 5.0, -1.5)  [Climbs Ramp 1->2 + switches to floor2_map]
  3. Floor 3 West Room     (-2.0, -1.5)  [Climbs Ramp 2->3 + switches to floor3_map]
  4. Floor 1 Home Origin   ( 0.0,  0.0)  [Descends Ramp 2->3 & Ramp 1->2 -> floor1_map]
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


WAYPOINTS = [
    ('floor1',  3.2,  1.8, 'Floor 1 - East Room'),
    ('floor2',  5.0, -1.5, 'Floor 2 - East Corridor'),
    ('floor3', -1.8, -4.8, 'Floor 3 - South-East Corner'),
    ('floor1',  0.0,  0.0, 'Floor 1 - Home Origin'),
]


class MultiFloorPatrol(Node):
    def __init__(self):
        super().__init__('multi_floor_patrol')
        self.pub_goal = self.create_publisher(PoseStamped, '/multi_floor_goal', 10)
        self.status = 'IDLE'
        self.status_seq = 0
        self.sub_status = self.create_subscription(
            String, '/floor_manager/mission_status', self.status_cb, 10
        )

    def status_cb(self, msg: String):
        if msg.data != self.status:
            self.status = msg.data
            self.status_seq += 1

    def run_patrol(self):
        self.get_logger().info('Waiting for floor_manager_node subscriber on /multi_floor_goal...')
        while rclpy.ok() and self.pub_goal.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(f'Starting 3-Story Multi-Floor Patrol ({len(WAYPOINTS)} waypoints)...')
        for idx, (floor, x, y, label) in enumerate(WAYPOINTS, start=1):
            self.get_logger().info(f'=== Waypoint {idx}/{len(WAYPOINTS)}: {label} ({floor}: {x:.1f}, {y:.1f}) ===')
            prev_seq = self.status_seq

            msg = PoseStamped()
            msg.header.frame_id = floor
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.orientation.w = 1.0
            self.pub_goal.publish(msg)

            seen_running = False
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.status == 'RUNNING' and self.status_seq > prev_seq:
                    seen_running = True
                if seen_running and self.status in ('SUCCEEDED', 'FAILED'):
                    break

            if self.status == 'SUCCEEDED':
                self.get_logger().info(f'Reached {label}! Pausing 2s before next waypoint...')
                time.sleep(2.0)
            else:
                self.get_logger().error(f'Mission FAILED at {label}. Stopping patrol.')
                return

        self.get_logger().info('FULL 3-STORY MULTI-FLOOR PATROL MISSION COMPLETED SUCCESSFULLY!')


def main(args=None):
    rclpy.init(args=args)
    node = MultiFloorPatrol()
    try:
        node.run_patrol()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
