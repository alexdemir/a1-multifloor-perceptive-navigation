#!/usr/bin/env python3
"""
Interactive Multi-Floor Goal Selector GUI with Target Floor Map Preview.

Features:
  - Select target floor (Floor 1 / 2 / 3) and preview its actual 2D occupancy map.
  - Forbidden ramp zones are shaded in red so goals cannot be placed on ramps.
  - Click directly on the GUI map preview OR use RViz 'Publish Point' (/clicked_point).
  - Validates every goal against ramp zones, walls, unknown space, and map bounds
    before dispatching to /multi_floor_goal.
"""

import math
import os
import threading
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class FloorGoalGuiNode(Node):
    def __init__(self, on_status_update, on_odom_update):
        super().__init__('floor_goal_gui')
        pkg_dir = get_package_share_directory('a1_description')
        default_topo = os.path.join(pkg_dir, 'config', 'building_topology.yaml')
        maps_dir = os.path.join(pkg_dir, 'maps')
        self.declare_parameter('topology_file', default_topo)
        topo_file = self.get_parameter('topology_file').value

        with open(topo_file, 'r') as f:
            topo = yaml.safe_load(f)

        self.connectors = topo.get('connectors', [])
        self.floors = sorted(
            [(name, float(cfg['z_center']), int(cfg['id']), cfg['map']) for name, cfg in topo['floors'].items()],
            key=lambda x: x[1],
        )
        self.floor_maps = {}
        for name, _, _, map_file in self.floors:
            mpath = map_file if os.path.isabs(map_file) else os.path.join(maps_dir, map_file)
            with open(mpath, 'r') as mf:
                mcfg = yaml.safe_load(mf)
            img_path = mcfg['image']
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(mpath), img_path)
            grid = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
            self.floor_maps[name] = {
                'grid': grid,
                'res': float(mcfg['resolution']),
                'ox': float(mcfg['origin'][0]),
                'oy': float(mcfg['origin'][1]),
            }

        self.selected_floor = self.floors[0][0]
        self.current_floor = self.floors[0][0]
        self.mission_status = 'IDLE'
        self.on_status_update = on_status_update
        self.on_odom_update = on_odom_update

        self.pub_goal = self.create_publisher(PoseStamped, '/multi_floor_goal', 10)
        self.sub_click = self.create_subscription(PointStamped, '/clicked_point', self.clicked_cb, 10)
        self.sub_floor = self.create_subscription(String, '/current_floor_name', self.floor_cb, 10)
        self.sub_status = self.create_subscription(String, '/floor_manager/mission_status', self.status_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odom_cb, 10)

    def floor_cb(self, msg: String):
        self.current_floor = msg.data
        self.on_status_update(self.current_floor, self.mission_status, None, False)

    def status_cb(self, msg: String):
        self.mission_status = msg.data
        is_err = msg.data.startswith('REJECTED') or msg.data == 'FAILED'
        self.on_status_update(self.current_floor, self.mission_status, msg.data if is_err else None, is_err)

    def odom_cb(self, msg: Odometry):
        self.on_odom_update(float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def validate_goal(self, floor_name: str, x: float, y: float):
        for conn in self.connectors:
            if conn.get('upper_floor') != floor_name:
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
                return False, f"Rejected: ({x:.2f}, {y:.2f}) is on {conn['name']} ramp"

        fm = self.floor_maps.get(floor_name)
        if fm is not None:
            grid = fm['grid']
            res, ox, oy = fm['res'], fm['ox'], fm['oy']
            h, w = grid.shape
            col = int((x - ox) / res)
            row = h - 1 - int((y - oy) / res)
            if not (0 <= col < w and 0 <= row < h):
                return False, f'Rejected: ({x:.2f}, {y:.2f}) outside map'
            val = int(grid[row, col])
            if val < 100:
                return False, f'Rejected: ({x:.2f}, {y:.2f}) inside wall'
            if val < 230:
                return False, f'Rejected: ({x:.2f}, {y:.2f}) not walkable on {floor_name}'
            r_px = max(1, int(round(0.25 / res)))
            patch = grid[max(0, row - r_px):min(h, row + r_px + 1), max(0, col - r_px):min(w, col + r_px + 1)]
            if np.any(patch < 100):
                return False, f'Rejected: ({x:.2f}, {y:.2f}) too close to wall'
        return True, 'OK'

    def send_goal(self, floor_name: str, x: float, y: float):
        ok, reason = self.validate_goal(floor_name, x, y)
        if not ok:
            self.get_logger().warn(reason)
            self.on_status_update(self.current_floor, self.mission_status, reason, True)
            return False

        z_center = next((z for n, z, _, _ in self.floors if n == floor_name), 0.0)
        msg = PoseStamped()
        msg.header.frame_id = floor_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z_center)
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)
        info = f'Sent -> {floor_name} ({x:.2f}, {y:.2f})'
        self.get_logger().info(info)
        self.on_status_update(self.current_floor, self.mission_status, info, False)
        return True

    def clicked_cb(self, msg: PointStamped):
        self.send_goal(self.selected_floor, msg.point.x, msg.point.y)


def build_floor_photo(fm, connectors, floor_name, scale=1.25):
    grid = fm['grid']
    h, w = grid.shape
    rgb = np.stack([grid, grid, grid], axis=-1)
    # Colorize walls dark blue-gray, unknown dark slate, free soft gray
    rgb[grid < 100] = [30, 32, 48]
    rgb[(grid >= 100) & (grid < 230)] = [69, 71, 90]
    rgb[grid >= 230] = [220, 224, 232]

    # Tint forbidden ramp rectangles in soft red
    res, ox, oy = fm['res'], fm['ox'], fm['oy']
    for conn in connectors:
        if conn.get('upper_floor') != floor_name:
            continue
        bx, by = float(conn['bottom_xy'][0]), float(conn['bottom_xy'][1])
        tx, ty = float(conn['top_xy'][0]), float(conn['top_xy'][1])
        dx, dy = tx - bx, ty - by
        length = math.hypot(dx, dy)
        if length < 0.5:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        ys_idx, xs_idx = np.mgrid[0:h, 0:w]
        wx = ox + (xs_idx + 0.5) * res
        wy = oy + (h - 1 - ys_idx + 0.5) * res
        s = (wx - bx) * ux + (wy - by) * uy
        d = (wx - bx) * nx + (wy - by) * ny
        mask = (s >= 0.75) & (s <= length - 0.55) & (np.abs(d) <= 1.15) & (grid >= 230)
        rgb[mask] = [235, 130, 140]

    pil_img = Image.fromarray(rgb, mode='RGB').resize(
        (int(round(w * scale)), int(round(h * scale))), Image.NEAREST
    )
    return ImageTk.PhotoImage(pil_img), int(round(w * scale)), int(round(h * scale))


def main(args=None):
    rclpy.init(args=args)

    root = tk.Tk()
    root.title('A1 Floor Selector')
    root.attributes('-topmost', True)
    root.resizable(False, False)
    root.configure(bg='#1e1e2e', padx=10, pady=8)

    status_var = tk.StringVar(value='Current Floor: -  |  Status: IDLE')
    last_var = tk.StringVar(value='Click map below or use RViz "Publish Point"')
    scale = 1.25
    robot_xy = [0.0, 0.0]

    last_label = None

    def on_update(cur_floor, m_status, last_msg, is_err):
        def _apply():
            status_var.set(f'Current Floor: {cur_floor}  |  Status: {m_status}')
            if last_msg:
                last_var.set(last_msg)
                if last_label is not None:
                    last_label.configure(fg='#f38ba8' if is_err else '#f9e2af')
        root.after(0, _apply)

    def on_odom(rx, ry):
        robot_xy[0] = rx
        robot_xy[1] = ry

    node = FloorGoalGuiNode(on_update, on_odom)

    tk.Label(
        root, text='Select Floor -> Click Map Preview (or RViz Publish Point)',
        fg='#cdd6f4', bg='#1e1e2e', font=('Sans', 9, 'bold')
    ).pack(pady=(0, 4))

    tk.Label(
        root, textvariable=status_var,
        fg='#a6e3a1', bg='#1e1e2e', font=('Sans', 9)
    ).pack(pady=(0, 6))

    btn_frame = tk.Frame(root, bg='#1e1e2e')
    btn_frame.pack(pady=(0, 6))

    # Build map images once
    floor_photos = {}
    cw, ch = 400, 350
    for name, _, _, _ in node.floors:
        photo, pw, ph = build_floor_photo(node.floor_maps[name], node.connectors, name, scale=scale)
        floor_photos[name] = photo
        cw, ch = pw, ph

    canvas = tk.Canvas(root, width=cw, height=ch, bg='#11111b', highlightthickness=1, highlightbackground='#45475a', cursor='crosshair')
    canvas.pack(pady=(0, 6))
    img_item = canvas.create_image(0, 0, anchor=tk.NW, image=floor_photos[node.selected_floor])
    robot_dot = canvas.create_oval(-10, -10, -10, -10, fill='#a6e3a1', outline='#11111b', width=1.5)
    goal_dot = canvas.create_oval(-10, -10, -10, -10, fill='#89b4fa', outline='#11111b', width=1.5)

    def world_to_canvas(floor_name, wx, wy):
        fm = node.floor_maps[floor_name]
        h = fm['grid'].shape[0]
        cx = ((wx - fm['ox']) / fm['res']) * scale
        cy = (h - 1 - ((wy - fm['oy']) / fm['res'])) * scale
        return cx, cy

    def canvas_to_world(floor_name, cx, cy):
        fm = node.floor_maps[floor_name]
        h = fm['grid'].shape[0]
        wx = fm['ox'] + (cx / scale) * fm['res']
        wy = fm['oy'] + (h - 1 - (cy / scale)) * fm['res']
        return wx, wy

    def on_canvas_click(event):
        wx, wy = canvas_to_world(node.selected_floor, event.x, event.y)
        if node.send_goal(node.selected_floor, wx, wy):
            canvas.coords(goal_dot, event.x - 5, event.y - 5, event.x + 5, event.y + 5)

    canvas.bind('<Button-1>', on_canvas_click)

    buttons = {}

    def select_floor(fname):
        node.selected_floor = fname
        canvas.itemconfigure(img_item, image=floor_photos[fname])
        canvas.coords(goal_dot, -10, -10, -10, -10)
        for name, btn in buttons.items():
            if name == fname:
                btn.configure(bg='#89b4fa', fg='#11111b')
            else:
                btn.configure(bg='#313244', fg='#cdd6f4')

    for name, z_val, fid, _ in node.floors:
        b = tk.Button(
            btn_frame, text=f'Floor {fid} ({z_val:.1f}m)', width=12, height=1,
            font=('Sans', 9, 'bold'), relief='flat', cursor='hand2', pady=4,
            command=lambda n=name: select_floor(n)
        )
        b.pack(side=tk.LEFT, padx=3)
        buttons[name] = b

    select_floor(node.floors[0][0])

    quick_points = {
        'floor1': (0.0, 0.0),
        'floor2': (4.0, -3.5),
        'floor3': (-1.8, -4.8),
    }
    tk.Button(
        root, text='Go to Selected Floor Center',
        bg='#45475a', fg='#cdd6f4', activebackground='#585b70',
        relief='flat', font=('Sans', 9), cursor='hand2', pady=4,
        command=lambda: node.send_goal(
            node.selected_floor,
            *quick_points.get(node.selected_floor, (0.0, 0.0))
        )
    ).pack(fill=tk.X, pady=(0, 5))

    last_label = tk.Label(
        root, textvariable=last_var,
        fg='#f9e2af', bg='#1e1e2e', font=('Sans', 8)
    )
    last_label.pack()

    def refresh_robot_dot():
        cx, cy = world_to_canvas(node.selected_floor, robot_xy[0], robot_xy[1])
        color = '#a6e3a1' if node.current_floor == node.selected_floor else '#6c7086'
        canvas.itemconfigure(robot_dot, fill=color)
        canvas.coords(robot_dot, cx - 4, cy - 4, cx + 4, cy + 4)
        root.after(150, refresh_robot_dot)

    root.after(150, refresh_robot_dot)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
