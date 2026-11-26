#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from ament_index_python.packages import get_package_share_directory

# 12 Joint Names - Ordered identically to Isaac Gym training (FL, FR, RL, RR)
JOINT_NAMES = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]

# Nominal standing joint angles trained in Isaac Gym (Radians) - (FL, FR, RL, RR)
DEFAULT_DOF_POS = np.array([
     0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
     0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5
], dtype=np.float32)

# Learned constant DC hip adduction bias from Isaac Gym training (where self_collisions=1 was disabled)
HIP_DC_BIAS = np.array([
    -0.02, 0.0, 0.0,   # FL_hip
    +0.12, 0.0, 0.0,   # FR_hip
    -0.22, 0.0, 0.0,   # RL_hip
    +0.17, 0.0, 0.0    # RR_hip
], dtype=np.float32)

# Symmetric, upright, and motionless standing joint angles on flat ground
STAND_DOF_POS = np.array([
     0.0, 0.80, -1.45,
     0.0, 0.80, -1.45,
     0.0, 0.85, -1.45,
     0.0, 0.85, -1.45
], dtype=np.float32)

def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

class RLLocomotionController(Node):
    def __init__(self):
        super().__init__('rl_locomotion_controller')

        # Load policy weights
        self.declare_parameter('policy_file', 'a1_policy_rough.npz')
        self.declare_parameter('use_elevation_map', True)
        policy_file = str(self.get_parameter('policy_file').value)
        self.use_elevation_map = bool(self.get_parameter('use_elevation_map').value)
        self.hip_bias = np.zeros(12, dtype=np.float32) if 'rough' in policy_file else HIP_DC_BIAS

        try:
            pkg_dir = get_package_share_directory('a1_description')
            model_path = policy_file if os.path.isabs(policy_file) else os.path.join(pkg_dir, 'config', policy_file)
        except Exception:
            model_path = f'/home/alex/a1_navigation_ws/src/a1_description/config/{policy_file}'

        if not os.path.exists(model_path):
            model_path = '/home/alex/a1_navigation_ws/src/a1_description/config/a1_policy.npz'
            self.hip_bias = HIP_DC_BIAS

        self.get_logger().info(
            f'Loading RL policy from: {model_path} '
            f'(elevation_map={self.use_elevation_map}, hip_bias={"off" if np.all(self.hip_bias == 0) else "on"})'
        )
        data = np.load(model_path)
        self.w0, self.b0 = data['w0'], data['b0']
        self.w1, self.b1 = data['w1'], data['b1']
        self.w2, self.b2 = data['w2'], data['b2']
        self.w3, self.b3 = data['w3'], data['b3']
        self.get_logger().info('RL policy loaded successfully.')

        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.sub_js = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.sub_models = self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        self.sub_height_scan = self.create_subscription(
            Float32MultiArray, '/elevation_map/height_scan', self.height_scan_callback, 10
        )

        # Publisher -> ros2_control JointGroupPositionController
        self.pub_joint_cmd = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        # Raw velocity command state
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.last_js_sim_ns = 0
        self.last_cmd_sim_ns = -10_000_000_000
        self.last_active_sim_ns = -10_000_000_000
        self.last_step_sim_ns = None
        self.step_dt_ns = 20_000_000  # exact 50 Hz in Gazebo sim time (dt = 0.020s)
        self.ever_moved = False

        # Smoothed (ramped) velocity commands
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.smooth_wz = 0.0

        self.cur_dof = STAND_DOF_POS.copy()
        self.cur_vel = np.zeros(12, dtype=np.float32)
        self.target_dof = STAND_DOF_POS.copy()
        self.last_action = np.zeros(12, dtype=np.float32)

        self.has_joint_states = False
        self.initialized_pose = False

        self.base_quat = [0.0, 0.0, 0.0, 1.0]
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.filtered_pitch = 0.0
        self.ramp_vx = 0.0
        self.target_yaw = 0.0
        self.yaw_initialized = False

        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.live_height_scan = np.full(187, -0.90, dtype=np.float32)
        self.last_height_scan_sim_ns = -10_000_000_000

        # Driven deterministically at 50 Hz sim time from /joint_states header.stamp (zero ROS timer jitter)
        self.get_logger().info('RL ros2_control locomotion node (50 Hz sim-time synced with Hill-Hold) is ready.')

    def elu(self, x):
        return np.where(x > 0, x, np.exp(np.clip(x, -20.0, 20.0)) - 1.0)

    def forward(self, x):
        h0 = self.elu(self.w0 @ x + self.b0)
        h1 = self.elu(self.w1 @ h0 + self.b1)
        h2 = self.elu(self.w2 @ h1 + self.b2)
        return self.w3 @ h2 + self.b3

    def get_slope_adapted_stand_pose(self):
        """
        Computes a slope-adaptive standing pose (Hill-Hold) to prevent tipping on ramps.
        Inactive on flat ground (|pitch| < 6 deg) and during initial spawn/standby.
        """
        if not self.ever_moved:
            return STAND_DOF_POS.copy()

        nose_up = float(np.clip(-self.filtered_pitch, -0.35, 0.35))
        if abs(nose_up) < 0.10:
            return STAND_DOF_POS.copy()

        pose = STAND_DOF_POS.copy()
        # Scale adaptation smoothly beyond the 0.10 rad (~6 deg) threshold
        slope_factor = math.copysign(abs(nose_up) - 0.10, nose_up)

        # Shift thighs into the incline: decreasing thigh angle leans legs forward (+X)
        thigh_shift = 0.50 * slope_factor
        pose[1] -= thigh_shift   # FL_thigh
        pose[4] -= thigh_shift   # FR_thigh
        pose[7] -= thigh_shift   # RL_thigh
        pose[10] -= thigh_shift  # RR_thigh

        # Crouch uphill knees and extend downhill knees to keep CoM centered
        pose[2] -= 0.60 * slope_factor   # FL_calf
        pose[5] -= 0.60 * slope_factor   # FR_calf
        pose[8] += 0.45 * slope_factor   # RL_calf
        pose[11] += 0.45 * slope_factor  # RR_calf
        return pose

    def height_scan_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 187:
            self.live_height_scan = np.asarray(msg.data, dtype=np.float32)
            self.last_height_scan_sim_ns = self.last_js_sim_ns

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vx = float(np.clip(msg.linear.x, -1.0, 1.0))
        self.cmd_vy = float(np.clip(msg.linear.y, -0.6, 0.6))
        self.cmd_wz = float(np.clip(msg.angular.z, -1.0, 1.0))
        self.last_cmd_sim_ns = self.last_js_sim_ns

    def imu_callback(self, msg: Imu):
        self.base_ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float32)

        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.base_quat = [qx, qy, qz, qw]

        # Current heading (yaw) and pitch angles
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        self.current_pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        self.filtered_pitch = 0.95 * self.filtered_pitch + 0.05 * self.current_pitch

        if not self.yaw_initialized:
            self.target_yaw = self.current_yaw
            self.yaw_initialized = True

        # Gravity vector projected into the robot base frame (compatible with Isaac Gym quat_rotate_inverse)
        gx = 2.0 * (qw * qy - qx * qz)
        gy = -2.0 * (qw * qx + qy * qz)
        gz = -(1.0 - 2.0 * (qx * qx + qy * qy))
        self.projected_gravity = np.array([gx, gy, gz], dtype=np.float32)

    def model_states_callback(self, msg: ModelStates):
        if 'a1_quadruped' in msg.name:
            idx = msg.name.index('a1_quadruped')
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            vz = msg.twist[idx].linear.z

            qx, qy, qz, qw = self.base_quat
            r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
            r01 = 2.0 * (qx * qy - qw * qz)
            r02 = 2.0 * (qx * qz + qw * qy)

            r10 = 2.0 * (qx * qy + qw * qz)
            r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
            r12 = 2.0 * (qy * qz - qw * qx)

            r20 = 2.0 * (qx * qz - qw * qy)
            r21 = 2.0 * (qy * qz + qw * qx)
            r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

            bx = r00 * vx + r10 * vy + r20 * vz
            by = r01 * vx + r11 * vy + r21 * vz
            bz = r02 * vx + r12 * vy + r22 * vz
            self.base_lin_vel = np.array([bx, by, bz], dtype=np.float32)

    def joint_state_callback(self, msg: JointState):
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.cur_dof[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.cur_vel[i] = msg.velocity[idx]

        if not self.initialized_pose:
            # Start target_dof from actual current joint positions (zero step-jerk / zero catapult flip)
            # and smoothly ramp up to STAND_DOF_POS over ~0.6s.
            self.target_dof = self.cur_dof.copy()
            self.initialized_pose = True
            self.has_joint_states = True
            self.get_logger().info('Gazebo joint states received. Smooth stand-up to symmetric pose active.')

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds
        self.last_js_sim_ns = stamp_ns

        if self.last_step_sim_ns is None:
            self.last_step_sim_ns = stamp_ns
            self.control_loop(stamp_ns)
        elif (stamp_ns - self.last_step_sim_ns) >= self.step_dt_ns:
            if (stamp_ns - self.last_step_sim_ns) > 2 * self.step_dt_ns:
                self.last_step_sim_ns = stamp_ns
            else:
                self.last_step_sim_ns += self.step_dt_ns
            self.control_loop(stamp_ns)

    def control_loop(self, sim_now_ns: int):
        if not self.has_joint_states:
            return

        cmd_age = (sim_now_ns - self.last_cmd_sim_ns) * 1e-9

        # Reset desired velocity to zero if no new command has arrived within 0.6 seconds
        if cmd_age < 0.6:
            desired_vx = self.cmd_vx
            desired_vy = self.cmd_vy
            desired_wz = self.cmd_wz
        else:
            desired_vx = 0.0
            desired_vy = 0.0
            desired_wz = 0.0

        # Smoothly ramp velocity commands
        self.smooth_vx = 0.80 * self.smooth_vx + 0.20 * desired_vx
        self.smooth_vy = 0.80 * self.smooth_vy + 0.20 * desired_vy
        self.smooth_wz = 0.80 * self.smooth_wz + 0.20 * desired_wz

        if abs(self.smooth_vx) < 0.02:
            self.smooth_vx = 0.0
        if abs(self.smooth_vy) < 0.02:
            self.smooth_vy = 0.0
        if abs(self.smooth_wz) < 0.02:
            self.smooth_wz = 0.0

        has_cmd = (abs(self.smooth_vx) > 0.02 or abs(self.smooth_vy) > 0.02 or abs(self.smooth_wz) > 0.02)
        if has_cmd:
            self.last_active_sim_ns = sim_now_ns
            self.ever_moved = True

        time_since_active = (sim_now_ns - self.last_active_sim_ns) * 1e-9

        # Slope-adapted standing posture (flat ground uses STAND_DOF_POS; ramps shift CoM over support polygon)
        adapted_stand_pose = self.get_slope_adapted_stand_pose()

        # Transition into Hill-Hold within 0.45s on steep slopes (> 6 deg) or 0.70s on flat ground
        on_slope = abs(self.filtered_pitch) > 0.10
        brake_duration = 0.45 if on_slope else 0.70
        is_braking_phase = self.ever_moved and (time_since_active < brake_duration)

        if not has_cmd and not is_braking_phase:
            # Robot is completely stopped: smoothly converge into standing posture without jerk
            self.target_yaw = self.current_yaw
            self.ramp_vx = 0.0
            alpha = 0.08 if not self.ever_moved else 0.15
            self.target_dof = (1.0 - alpha) * self.target_dof + alpha * adapted_stand_pose
            if np.max(np.abs(self.target_dof - adapted_stand_pose)) < 0.002:
                self.target_dof = adapted_stand_pose.copy()
            self.last_action = np.zeros(12, dtype=np.float32)
        else:
            body_yaw_rate = abs(float(self.base_ang_vel[2]))
            if abs(self.smooth_wz) > 0.03 or (not has_cmd and body_yaw_rate > 0.10):
                self.target_yaw = self.current_yaw
                effective_wz = self.smooth_wz
            else:
                yaw_error = wrap_to_pi(self.target_yaw - self.current_yaw)
                effective_wz = float(np.clip(2.5 * yaw_error, -1.0, 1.0))

            nose_up = float(np.clip(-self.filtered_pitch, -0.35, 0.35))
            if abs(self.smooth_vy) < 0.03 and abs(self.smooth_vx) > 0.05:
                is_symmetric_policy = np.all(self.hip_bias == 0)
                if abs(nose_up) > 0.10:
                    if is_symmetric_policy:
                        effective_vy = float(np.clip(-0.65 * self.base_lin_vel[1], -0.4, 0.4))
                    else:
                        effective_vy = float(np.clip(-0.09 * self.smooth_vx - 0.75 * self.base_lin_vel[1], -0.4, 0.4))
                elif is_symmetric_policy:
                    effective_vy = float(np.clip(-0.55 * self.base_lin_vel[1], -0.4, 0.4))
                else:
                    effective_vy = float(np.clip(-0.04 * self.smooth_vx - 0.55 * self.base_lin_vel[1], -0.4, 0.4))
            else:
                effective_vy = self.smooth_vy

            # Smooth slope-aware Ramp Climb Mode (flat ground untouched):
            #   - Uphill (nose_up > 6 deg): cap |wz| <= 0.45 rad/s and allow steady 0.60-0.70 m/s
            #   - Downhill (nose_up < -10.3 deg): cap at 0.42 m/s + |wz| <= 0.50 rad/s (ignores normal +5..8 deg trot pitch)
            if nose_up > 0.10 and self.smooth_vx > 0.05:
                effective_wz = float(np.clip(effective_wz, -0.45, 0.45))
                if self.smooth_vx > 0.15 and abs(effective_wz) < 0.18:
                    target_vx = float(np.clip(max(self.smooth_vx, 0.65), 0.0, 0.70))
                elif self.smooth_vx > 0.15:
                    target_vx = float(np.clip(self.smooth_vx, 0.0, 0.60))
                else:
                    target_vx = self.smooth_vx
            elif nose_up < -0.18 and self.smooth_vx > 0.05:
                effective_wz = float(np.clip(effective_wz, -0.50, 0.50))
                target_vx = float(np.clip(self.smooth_vx, 0.0, 0.42))
            else:
                target_vx = self.smooth_vx

            self.ramp_vx = 0.80 * self.ramp_vx + 0.20 * target_vx
            effective_vx = self.ramp_vx

            virtual_dof = self.cur_dof + self.hip_bias

            # Use live 2.5D Elevation Map height_scan when fresh (< 0.5s sim time), else flat fallback (-0.90)
            using_live_elev = self.use_elevation_map and (sim_now_ns - self.last_height_scan_sim_ns) < 500_000_000
            if using_live_elev:
                height_scan_obs = self.live_height_scan
            else:
                height_scan_obs = np.full(187, -0.90, dtype=np.float32)

            # Isaac Gym observation vector (235 dimensions)
            obs = np.concatenate([
                self.base_lin_vel * 2.0,
                self.base_ang_vel * 0.25,
                self.projected_gravity,
                np.array([effective_vx * 2.0, effective_vy * 2.0, effective_wz * 0.25], dtype=np.float32),
                (virtual_dof - DEFAULT_DOF_POS) * 1.0,
                self.cur_vel * 0.05,
                self.last_action,
                height_scan_obs
            ]).astype(np.float32)

            # RL policy inference
            raw_action = self.forward(obs)
            self.last_action = np.clip(raw_action, -100.0, 100.0)

            rl_target = (DEFAULT_DOF_POS + 0.25 * self.last_action) - self.hip_bias

            # Manual slope_feedforward is only needed in blind fallback mode;
            # perceptive symmetric policy lifts legs cleanly on its own without hill-hold bias fighting it.
            if not using_live_elev:
                if nose_up > 0.10:
                    rl_target = rl_target + (adapted_stand_pose - STAND_DOF_POS) * 0.65
                elif nose_up < -0.18:
                    rl_target = rl_target + (adapted_stand_pose - STAND_DOF_POS) * 0.25

            # Smoothly blend from RL target into slope-adapted standing pose during the braking window
            if not has_cmd and is_braking_phase:
                blend = float(np.clip(time_since_active / brake_duration, 0.0, 1.0))
                rl_target = (1.0 - blend) * rl_target + blend * adapted_stand_pose

            self.target_dof = rl_target

        cmd_msg = Float64MultiArray()
        cmd_msg.data = [float(v) for v in self.target_dof]
        self.pub_joint_cmd.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLLocomotionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
