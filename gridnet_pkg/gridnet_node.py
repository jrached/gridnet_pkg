import rclpy 
from rclpy.node import Node 
import numpy as np 
import struct 
import torch 
from scipy.spatial.transform import Rotation as R 
from math import ceil 

from geometry_msgs.msg import Pose, PoseStamped, TwistStamped, PoseArray   
from sensor_msgs.msg import PointCloud2 
from scipy.ndimage import label

from .gridnet import GridNet 

DEVICE = 'cuda:0' 

class GridNetNode(Node): 
    def __init__(self): 
        super().__init__('gridnet_node')
        
        self.get_logger().info("\nRunning GridNet")

        # Subscriptions
        self.pc_sub_ = self.create_subscription(PointCloud2, "cloud_topic", self.pc_cb_, 10)
        self.pose_sub = self.create_subscription(PoseStamped, "pose_topic", self.pose_cb_, 10)
        self.twist_sub = self.create_subscription(TwistStamped, "twist_topic", self.twist_cb_, 10)

        # Publishers 
        self.pose_arr_pub_ = self.create_publisher(PoseArray, "est_obs_poses", 10) 

        # Class variables 
        self.ego_pose = None
        self.ego_twist = None
        self.max_points, self.seq_len, self.d = 21000, 3, 4 
        self.grid_dims = [16, 16]
        self.voxel_size = 0.2 
        self.state_dim = 6

        # Tensors 
        num_voxels_x, num_voxels_y = int(self.grid_dims[0] / self.voxel_size), int(self.grid_dims[1] / self.voxel_size) 
        self.in_scan_seq = torch.zeros((1, self.seq_len, 1, self.max_points, self.d)) # (B, T, C, N) 
        self.in_scan_seq_copy = torch.zeros((1, self.seq_len, 1, self.max_points, self.d)) # (B, T, C, N) 
        self.state_seq = torch.zeros(((1, self.seq_len, self.state_dim))) # (B, T, xyz + quat)
        self.state_seq_copy = torch.zeros(((1, self.seq_len, self.state_dim))) # (B, T, xyz + quat)
        self.targ_grid_seq = torch.zeros((1, self.seq_len, 1, num_voxels_y, num_voxels_x)) # (B, T, C, H, W)

        # Model 
        in_dim, hidden_dim = 1152, 786 
        path_to_model = "/Volume/gridnet_ws/src/gridnet_pkg/best_model/gridnet_10hz_47val.pt"
        self.model = GridNet(seq_length=self.seq_len, in_dim=in_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
        self.model.to(DEVICE) 
        self.model.eval() 

    def pose_cb_(self, msg): 
        pos = msg.pose.position
        att = msg.pose.orientation 
        self.ego_pose = np.array([[pos.x, pos.y, pos.z, att.x, att.y, att.z, att.w]])
    
    def twist_cb_(self, msg): 
        lin = msg.twist.linear
        ang = msg.twist.angular
        self.ego_twist = np.array([[lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]])

    def pc_cb_(self, msg): 
        with torch.no_grad():
            if self.ego_pose is not None and self.ego_twist is not None: 
                start_time = self.get_clock().now().nanoseconds 

                # Get point cloud points 
                pc_points = self.pc_points_from_rosbag_(msg)
                pc_points = torch.from_numpy(pc_points).to(device=DEVICE).float() 
                n, d = pc_points.shape 

                # Get state 
                local_twist = self.global_to_local_twist(self.ego_pose, self.ego_twist)
                local_twist = torch.from_numpy(local_twist).to(device=DEVICE).float() 
                state = local_twist 

                # Update pointcloud sliding window 
                self.in_scan_seq[:, 2, :, :n, :] = pc_points.view(1, 1, n, d) 
                self.in_scan_seq[:, 1] = self.in_scan_seq_copy[:, 2]
                self.in_scan_seq[:, 0] = self.in_scan_seq_copy[:, 1]
                self.in_scan_seq_copy = self.in_scan_seq.clone() 

                # Update state sliding window
                self.state_seq[:, 2] = state.view(1, self.state_dim)
                self.state_seq[:, 1] = self.state_seq_copy[:, 2]
                self.state_seq[:, 0] = self.state_seq_copy[:, 1]
                self.state_seq_copy = self.state_seq.clone() 

                # Estimate dynamic obstacle positions 
                sequence = {'input': (self.in_scan_seq, self.state_seq), 'target': self.targ_grid_seq}
                _, pred = self.model.loss(sequence) 
                obs_coords = self.get_poses_from_img_(pred, self.ego_pose)  

                # Populate pose array message and publish 
                obs_poses = self.populate_pose_arr_(obs_coords) 
                self.pose_arr_pub_.publish(obs_poses) 


    def pc_points_from_rosbag_(self, pc): 
        """
        Recovers pointcloud 3D points from ros message encoding.

        Input: 
            - Deserialized ros sensor_msgs/PointCloud2 message extracted from rdp GeneralData object, pc

        Output: 
            - A numpy array containing the 3D points and intensities of the pointcloud  
            
        """
        offsets = {f.name: f.offset for f in pc.fields}

        x_off = offsets['x']
        y_off = offsets['y']
        z_off = offsets['z']
        i_off = offsets['intensity']

        points = [] 
        fmt = '>f' if pc.is_bigendian else '<f'
        for r in range(pc.height): 
            row_base = r * pc.row_step
            for c in range(pc.width): 
                base = row_base + c * pc.point_step 
                x = struct.unpack_from(fmt, pc.data, base + x_off)[0]
                y = struct.unpack_from(fmt, pc.data, base + y_off)[0]
                z = struct.unpack_from(fmt, pc.data, base + z_off)[0]
                i = struct.unpack_from(fmt, pc.data, base + i_off)[0]
                points.append((x,y,z, i))
            
        return np.asarray(points)
    
    def get_poses_from_img_(self, pred, pose): 
        # Convert prediction into binary image 
        batch_entry = 0
        img = pred[batch_entry]
        mask = img > 0 
        img[mask] = 1 
        img[~mask] = 0        

        # Compute base to world transform 
        btw = np.zeros((4, 4))
        btw[:3, 3] = pose[:, :3].reshape(-1)
        btw[3, 3] = 1
        rot = R.from_quat(pose[:, 3:7])
        rot_matrix = rot.as_matrix()
        btw[:3, :3] = rot_matrix 
        btw = torch.from_numpy(btw).to(device=DEVICE).float()

        # Get dynamic obstacle pixel coords
        binary = img[0].detach().cpu().numpy().astype(np.uint8)
        labeled, num_objects = label(binary) 
        centroids = [] 
        for obj_id in range(1, num_objects+1):
             ys, xs = np.where(labeled == obj_id) 
             cx = xs.mean()
             cy = ys.mean() 
             centroids.append([cx, cy])

        if len(centroids) == 0:
            pred_coords = torch.empty((0, 2), device=DEVICE, dtype=torch.float32)
        else:
            pred_coords = torch.tensor(centroids, device=DEVICE, dtype=torch.float32)

        # Convert from grid to ego frame coords
        _, num_voxels_y, num_voxels_x = img.shape
        world_offset = torch.tensor([[num_voxels_x, num_voxels_y]], device=DEVICE) // 2 
        pred_coords -= world_offset
        pred_coords = pred_coords * self.voxel_size

        # Transform coords to global frame 
        pad = torch.zeros((pred_coords.shape[0], 2), device=DEVICE)
        pred_coords = torch.hstack((pred_coords, pad))
        pred_coords[:, -1] = 1
        world_coords = (btw @ pred_coords.T).T[:, :-2]

        return world_coords 
    
    def populate_pose_arr_(self, coords): 
        pose_arr = PoseArray() 
        pose_arr.header.stamp = self.get_clock().now().to_msg() 
        pose_arr.header.frame_id = "world" 

        poses = []
        for coord in coords: 
            pose = Pose() 
            pose.position.x = coord[0].item()
            pose.position.y = coord[1].item()
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            
            poses.append(pose)

        pose_arr.poses = poses 
        return pose_arr 
    
    def global_to_local_twist(self, pose, twist): 

        btw = np.zeros((3, 3))
        rot = R.from_quat(pose[0, 3:7])
        rot_matrix = rot.as_matrix()
        btw[:3, :3] = rot_matrix 
        wtb = btw.T

        linear_vel = (wtb @ twist[:, :3].T).T
        twist[:, :3] = linear_vel 
        return twist


def main(args=None):
    print('Running GridNet Node.')

    rclpy.init(args=args)

    gridnet_node = GridNetNode()
    rclpy.spin(gridnet_node)
    gridnet_node.destroy_node()
    rclpy.shutdown()     



if __name__ == '__main__':
    main()
