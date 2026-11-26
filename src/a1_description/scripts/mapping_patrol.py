#!/usr/bin/env python3
"""
Autonomous Full-Building 3-Story Mapping Patrol.

Drives the quadruped autonomously through every room, corridor, and ramp across
Floor 1, Floor 2, and Floor 3 while DLIO mapping (dlio_map_node) is active, then
automatically calls /save_pcd to export the complete 3-story dlio_map.pcd.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from direct_lidar_inertial_odometry.srv import SavePCD


MAPPING_WAYPOINTS = [
    # Floor 1: Full coverage (NW room, W hall, NE under-balcony, SE under-balcony)
    ('floor1', -4.5,  4.0, 'Floor 1 - North-West Room'),
    ('floor1', -4.5, -1.0, 'Floor 1 - West Corridor'),
    ('floor1',  4.5,  3.5, 'Floor 1 - North-East Room'),
    ('floor1',  4.5, -3.5, 'Floor 1 - South-East Room'),
    # Floor 2: Climbs Ramp 1->2, covers all 4 quadrants around F2 divider
    ('floor2',  2.0, -4.5, 'Floor 2 - South-West Landing'),
    ('floor2',  5.2, -3.5, 'Floor 2 - South-East Room'),
    ('floor2',  5.2,  3.5, 'Floor 2 - North-East Room'),
    ('floor2',  2.0,  1.2, 'Floor 2 - West Balcony'),
    # Floor 3: Climbs Ramp 2->3, covers landing & all corners around F3 divider
    ('floor3', -5.3,  3.5, 'Floor 3 - Ramp Arrival Deck'),
    ('floor3', -5.3, -4.0, 'Floor 3 - South-West Corner'),
    ('floor3', -2.0, -4.0, 'Floor 3 - South-East Corner'),
    ('floor3', -2.0,  1.0, 'Floor 3 - North-East Corner'),
    # Return to Floor 1 origin (descends Ramp 2->3 & Ramp 1->2)
    ('floor1',  0.0,  0.0, 'Floor 1 - Home Origin'),
]


class MappingPatrol(Node):
    def __init__(self):
        super().__init__('mapping_patrol')
        self.pub_goal = self.create_publisher(PoseStamped, '/multi_floor_goal', 10)
        self.status = 'IDLE'
        self.status_seq = 0
        self.sub_status = self.create_subscription(
            String, '/floor_manager/mission_status', self.status_cb, 10
        )
        self.cli_save_pcd = self.create_client(SavePCD, '/save_pcd')

    def status_cb(self, msg: String):
        if msg.data != self.status:
            self.status = msg.data
            self.status_seq += 1

    def save_map(self):
        self.get_logger().info('Calling /save_pcd to save complete 3-story dlio_map.pcd...')
        if not self.cli_save_pcd.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/save_pcd service not available (was mapping:=true set?).')
            return
        req = SavePCD.Request()
        req.leaf_size = 0.05
        req.save_path = '/home/alex/a1_navigation_ws/src/a1_description/maps'
        fut = self.cli_save_pcd.call_async(req)
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('3-story dlio_map.pcd saved! Now run pcd_to_grid_map.py to slice maps.')

    def run(self):
        self.get_logger().info('Waiting for floor_manager_node on /multi_floor_goal...')
        while rclpy.ok() and self.pub_goal.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.2)

        total = len(MAPPING_WAYPOINTS)
        self.get_logger().info(f'Starting 3-Story Autonomous Mapping Patrol ({total} waypoints)...')
        for idx, (floor, x, y, label) in enumerate(MAPPING_WAYPOINTS, start=1):
            self.get_logger().info(f'=== [{idx}/{total}] {label} ({floor}: {x:+.1f}, {y:+.1f}) ===')
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
                self.get_logger().info(f'Reached {label}.')
                time.sleep(1.0)
            else:
                self.get_logger().error(f'Waypoint failed at {label}. Continuing to next waypoint...')
                time.sleep(1.0)

        self.save_map()
        self.get_logger().info('3-STORY MAPPING PATROL COMPLETE!')


def main(args=None):
    rclpy.init(args=args)
    node = MappingPatrol()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
