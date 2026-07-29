#!/usr/bin/env python3

import math
import struct

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped TF conversions.
import tf2_ros
from vision_msgs.msg import Detection2DArray


class BottlePositionNode(Node):
    def __init__(self):
        super().__init__('bottle_position_node')

        self.declare_parameter('target_class_id', '39')
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('detections_topic', '/detections_output')
        self.declare_parameter('depth_image_topic', '/Orbbec/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/Orbbec/depth/camera_info')
        self.declare_parameter('position_topic', '/bottle/position')
        self.declare_parameter('search_radius', 4)

        self.target_class_id = (
            self.get_parameter('target_class_id').get_parameter_value().string_value
        )
        self.target_frame = (
            self.get_parameter('target_frame').get_parameter_value().string_value
        )
        detections_topic = (
            self.get_parameter('detections_topic').get_parameter_value().string_value
        )
        depth_image_topic = (
            self.get_parameter('depth_image_topic').get_parameter_value().string_value
        )
        camera_info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value
        )
        position_topic = (
            self.get_parameter('position_topic').get_parameter_value().string_value
        )
        self.search_radius = (
            self.get_parameter('search_radius').get_parameter_value().integer_value
        )

        self.latest_depth_image = None
        self.latest_camera_info = None

        self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            10,
        )

        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.create_subscription(
            Detection2DArray,
            detections_topic,
            self.detection_callback,
            10,
        )

        self.pub = self.create_publisher(
            PoseStamped,
            position_topic,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            'Publishing bottle positions from '
            f'{detections_topic}, {depth_image_topic}, and {camera_info_topic} '
            f'to {position_topic}; '
            f'class_id={self.target_class_id}, target_frame={self.target_frame}'
        )

    def depth_image_callback(self, msg):
        self.latest_depth_image = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def detection_callback(self, msg):
        if self.latest_depth_image is None or self.latest_camera_info is None:
            return

        bottle_det = None
        best_score = 0.0

        for det in msg.detections:
            if not det.results:
                continue

            result = det.results[0]
            class_id = result.hypothesis.class_id
            score = result.hypothesis.score

            if class_id == self.target_class_id and score > best_score:
                bottle_det = det
                best_score = score

        if bottle_det is None:
            return

        u = int(bottle_det.bbox.center.position.x)
        v = int(bottle_det.bbox.center.position.y)

        p = self.project_nearest_valid_depth(
            self.latest_depth_image,
            self.latest_camera_info,
            u,
            v,
        )
        if p is None:
            self.get_logger().warn(
                f'No valid depth near pixel ({u}, {v})',
                throttle_duration_sec=1.0,
            )
            return

        x, y, z = p

        pose_cam = PoseStamped()
        pose_cam.header = self.latest_depth_image.header
        pose_cam.pose.position.x = float(x)
        pose_cam.pose.position.y = float(y)
        pose_cam.pose.position.z = float(z)
        pose_cam.pose.orientation.w = 1.0

        try:
            pose_world = self.tf_buffer.transform(
                pose_cam,
                self.target_frame,
                timeout=Duration(seconds=0.1),
            )
            self.pub.publish(pose_world)

            self.get_logger().info(
                f'bottle {self.target_frame}: '
                f'x={pose_world.pose.position.x:.3f}, '
                f'y={pose_world.pose.position.y:.3f}, '
                f'z={pose_world.pose.position.z:.3f}, '
                f'score={best_score:.2f}'
            )

        except Exception as e:
            self.get_logger().warn(
                f'TF transform failed: {e}',
                throttle_duration_sec=1.0,
            )

    def project_nearest_valid_depth(
        self,
        image: Image,
        camera_info: CameraInfo,
        u: int,
        v: int,
    ):
        depth_sample = self.read_nearest_valid_depth(image, u, v)
        if depth_sample is None:
            return None

        depth_u, depth_v, z = depth_sample
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        if fx == 0.0 or fy == 0.0:
            return None

        x = (depth_u - cx) * z / fx
        y = (depth_v - cy) * z / fy
        return x, y, z

    def read_nearest_valid_depth(self, image: Image, u: int, v: int):
        depth = self.read_depth_from_image(image, u, v)
        if depth is not None:
            return u, v, depth

        for radius in range(1, self.search_radius + 1):
            for dv in range(-radius, radius + 1):
                for du in range(-radius, radius + 1):
                    if abs(du) != radius and abs(dv) != radius:
                        continue

                    depth_u = u + du
                    depth_v = v + dv
                    depth = self.read_depth_from_image(image, depth_u, depth_v)
                    if depth is not None:
                        return depth_u, depth_v, depth

        return None

    def read_depth_from_image(self, image: Image, u: int, v: int):
        if u < 0 or v < 0 or u >= image.width or v >= image.height:
            return None

        encoding = image.encoding.upper()
        if encoding in ['16UC1', 'MONO16']:
            offset = v * image.step + u * 2
            fmt = '>H' if image.is_bigendian else '<H'
            scale = 0.001
        elif encoding == '32FC1':
            offset = v * image.step + u * 4
            fmt = '>f' if image.is_bigendian else '<f'
            scale = 1.0
        else:
            self.get_logger().warn(
                f'Unsupported depth encoding: {image.encoding}',
                throttle_duration_sec=5.0,
            )
            return None

        try:
            depth = struct.unpack_from(fmt, image.data, offset)[0] * scale
        except Exception:
            return None

        if not math.isfinite(depth) or depth <= 0.0:
            return None

        return depth


def main():
    rclpy.init()
    node = BottlePositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
