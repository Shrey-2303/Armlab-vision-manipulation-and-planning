#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QPoint
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.intrinsic_matrix = np.eye(3, dtype=float)
        self.extrinsic_matrix = np.eye(4, dtype=float)
        self.distortion_matrix = np.zeros((5,1), dtype=float)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        
        self.cal_offset = [130, 50]
        
        """ grid points info """
        self.grid_x_points = np.arange(self.cal_offset[0], 1150, 50)
        self.grid_y_points = np.arange(self.cal_offset[1], 750, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        # Generate (x, y) coordinate pairs from meshgrid
        self.grid_coords = np.vstack([self.grid_points[0].ravel(), self.grid_points[1].ravel()]).T
        
        """ block info """        
        self.block_detections = {}     # {'Color': [{'is_large': T/F, 'centroid':(x,y), 'size': (a, b), 'rot': angle}]}
        self.irregular_detections = [] #[{'centroid':(x,y), 'size': (a, b), 'rot': angle}]
        self.min_contour_area = 300    # reduce the noise
        self.max_contour_area = 4000  
        self.size_threshold = 1000     # determine small or large
        self.drop_spots = np.array([
            []
        ])
        # self.block_contours = np.array([])
        # self.block_detections = np.array([])
        
        """ tag info """
        self.MIN_CALIBRATE_POINT = 4
        # Location of the 4 tags (Pixel 2D points)
        self.DEST_PTS = np.array([
            [250, 500], #1
            [750, 500], #2
            [750, 200], #3 
            [250, 200], #4
        ], dtype=float)
        self.DEST_PTS += self.cal_offset
        # Location of the 4 tags (World 3D points)
        self.DEST_PTS_WORLD = np.array([
            [-250, -25, 0],   #1  
            [250, -25, 0],    #2
            [250, 275, 0],    #3
            [-250, 275, 0],   #4
            [375, 400, 150],  #5
            [0, 425, 62],  #6
            [-375, 100, 150], #7
            [375, 100, 150]    #8
        ], dtype=float)
        # AprilTag location { TagID: [pixel.x, pixel.y] }
        self.apriltag_points = {}
        
        """ homography info """
        self.H = None
        self.cameraCalibrated = False
        self.board_dimension = [1000, 650]
        self.arm_offest = [[425, 400], [450, 550]]
        self.arm_dimension = [[150, 150], [100, 100]]
        
        # self.tag_detections = np.array([])
        # self.tag_locations = [[-250, -25], [250, -25], [250, 275]]

    def ColorizeDepthFrame(self):
        """!
        @brief      Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                        cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    """  Convert Qt to Frame  """
    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        
        return cv2.getAffineTransform(pts1, pts2)

    """  Load Files  """
    def loadCameraCalibration(self, file_intrinsic, file_distortion=None):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        self.intrinsic_matrix = np.loadtxt(file_intrinsic, delimiter=",")
        if file_distortion is not None:
            self.distortion_matrix = np.loadtxt(file_distortion, delimiter=",")
        else:
            self.distortion_matrix = None
        
    """  Block  """
    def addBlockInfo(self, img):
        """!
        @brief  adds bounding box detections for the video frame
        """
        for color in self.block_detections:
            for detection in self.block_detections[color]:
                ((x, y), (w, h), rot) = (detection["centroid"], detection["size"], detection["rot"])
                box = cv2.boxPoints(((x, y), (w, h), rot))
                box = np.int0(box)
                cv2.drawContours(img,[box],0,(0,0,255),2)
                if detection["is_large"]:
                    cv2.putText(img, "large", (int(x+w/2+5), int(y+h/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    # cv2.circle(img, (int(x), int(y)), 30, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "small", (int(x+w/2+5), int(y+h/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    # cv2.circle(img, (int(x), int(y)), 30, (0, 255, 0), 2)
                    
                # Bounding Box and Text
                cv2.putText(img, color, (int(x+w/2+5), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "angle: "+ str(int(rot)), (int(x+w/2+5), int(y+h*2/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        for detection in self.irregular_detections:
            ((x, y), (w, h), rot) = (detection["centroid"], detection["size"], detection["rot"])
            box = cv2.boxPoints(((x, y), (w, h), rot))
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,0),2)
            
    def processColorMask(self, img, mask_img, color_range, roi):
        """!
        @brief  mask the color range and return detection blocks
        """
        detections = {}
        irregular_detections = []
        for color, (lower, upper) in color_range.items():
            mask = cv2.inRange(mask_img, lower, upper)
            mask = cv2.bitwise_and(roi, mask)

            # Reduce Noise
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Opening
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closing
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            detections[color] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area and area < self.max_contour_area:
                    ((x, y), (w, h), rot) = cv2.minAreaRect(contour)
                    is_large = area > self.size_threshold
                    
                    # Draw whole contours
                    # cv2.drawContours(img, [contour], 0, (0, 255, 0), 1)
                    if h > w*1.5 or w > h*1.5:
                        #check if the detection is irregular
                        irregular_detections.append({"centroid": (x, y), "size": (w, h), 'rot': rot})
                    else:
                        detections[color].append({"is_large": is_large, "centroid": (x, y), "size": (w, h), 'rot': rot})

        return detections, irregular_detections
    
    def setAndExcludeROI(self, img, roi, offset, dimension, isExcludeRegion=False, text=''):
        """!
        @brief      Region of interest, include and exclude
        """
        if isExcludeRegion:
            roi = roi
        else:
            roi = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        
        if self.cameraCalibrated: 
            x, y = offset[0], offset[1]
            width, height = dimension[0], dimension[1]
            if isExcludeRegion:
                roi_mask = np.ones_like(roi) * 255
                roi_mask[y:y+height, x:x+width] = 0
            else:
                roi_mask = np.zeros_like(roi)
                roi_mask[y:y+height, x:x+width] = 255
            roi = cv2.bitwise_and(roi, roi_mask)
            cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 1)
            cv2.putText(img, text, (x+5, y+height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        return img, roi
    
    def blockDetector(self, img):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        #clear irregular_detections
        self.irregular_detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        
        # Color Ranges
        color_ranges_hsv = {
            'Blue': (np.array([94, 98, 40]), np.array([107, 255, 255])),
            'Green': (np.array([60, 70, 35]), np.array([100, 240, 140])),    # Done
            'Yellow': (np.array([17, 143, 113]), np.array([47, 255, 255])),  # Done
            'Orange': (np.array([0, 105, 90]), np.array([19, 255, 255])),
            'Violet': (np.array([105, 35, 0]), np.array([155, 160, 128])), 
        }
        color_ranges_ycrcb = {
            'Red': (np.array([10, 150, 106]), np.array([100, 255, 155])),
            # 'Violet': (np.array([22, 113, 132]), np.array([86, 145, 255])), 
        }
        
        # Region of Interest (ROI) - only the board(Not the arm)
        img, roi = self.setAndExcludeROI(img, img, self.cal_offset, self.board_dimension)
        arm_offset = [self.cal_offset[0]+self.arm_offest[0][0], self.cal_offset[1]+self.arm_offest[0][1]]
        img, roi = self.setAndExcludeROI(img, roi, arm_offset, self.arm_dimension[0], isExcludeRegion=True)
        arm_offset = [self.cal_offset[0]+self.arm_offest[1][0], self.cal_offset[1]+self.arm_offest[1][1]]
        img, roi = self.setAndExcludeROI(img, roi, arm_offset, self.arm_dimension[1], isExcludeRegion=True, text="RobotArm")
            
        # Process masks
        detections_ycrcb, iregdet = self.processColorMask(img, ycrcb, color_ranges_ycrcb, roi)
        detections_hsv, iregdet2 = self.processColorMask(img, hsv, color_ranges_hsv, roi)
        self.block_detections = {**detections_ycrcb, **detections_hsv}
        self.irregular_detections.extend(iregdet)
        self.irregular_detections.extend(iregdet2)
        
        # Add text to blocks
        self.addBlockInfo(img)
        
        return img

    def detectBlocksInDepthImage(self, img):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """        
        
    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        
        grid_image = self.TagImageFrame.copy()
        if self.cameraCalibrated:
            grid_color = (255, 255, 0)
            for x, y in self.grid_coords:
                cv2.circle(grid_image, (x, y), 2, grid_color, -1)
        
        self.GridFrame = grid_image
    
    """  Challenge  """
    def find_distance(self, block1, block2):
        return np.linalg.norm(block1[:2] - block2[:2], axis=1)
    
    def find_possible_locations(self, locations, occupied_blocks, threshold_distance):
        possible_locations = []

        for location in locations:
            distances = np.linalg.norm(location[:2] - occupied_blocks[:, :2], axis=1)
            if all(dist > threshold_distance for dist in distances):
                possible_locations.append(location)

        return possible_locations

    def find_place_location(self, occupied_blocks, task=''):
        if task=='task_1':
            large_locations = np.array([
                [320, -130, 0], [320, -50, 0],  
                [270, -130, 0], [270, -50, 0],  
                [220, -130, 0], [220, -50, 0],  
                [170, -130, 0], [170, -50, 0], 
                [120, -130, 0], [120, -50, 0], 
            ])
            small_locations = np.array([
                [-275, -150, 0], [-275, -100, 0], 
                [-225, -150, 0],  [-225, -100, 0],  
                [-175, -150, 0],  [-175, -100, 0],  
                [-125, -150, 0],  [-125, -100, 0],               
                [-125, -50, 0], [-125, -100, 0],
                [-175, -50, 0], [-175, -100, 0],
            ])
        if task=='task_4':
            large_locations = np.array([])
            small_locations = np.array([])
        
        threshold_distance_large = 45
        threshold_distance_small = 30

        occupied_blocks = np.array(occupied_blocks)
        large_place_location = self.find_possible_locations(large_locations, occupied_blocks, threshold_distance_large)
        small_place_location = self.find_possible_locations(small_locations, occupied_blocks, threshold_distance_small)

        return large_place_location, small_place_location

    """  AprilTag  """
    def get_apriltag_locations(self):
        real_locations = np.array(list(self.apriltag_points.values()))
        
        return self.H ,real_locations
    
    def findExtrinsicWithAprilTag(self):
        """
        @brief      Calibrate with the April Tags with 6 points, and show image
        """
        # Not enough points
        if not set([1,2,3,4,5,6,7,8]).issubset(set(self.apriltag_points)):
            print("INSUFFICIENT APRILTAG POINTS FOR SOLVEPNP")
            return False
        
        # Enough points (7 points)
        src_pts = []
        for i in [1,2,3,4,5,6,7,8]:
            src_pts.append(self.apriltag_points[i])

        # Parameters for solvePnP
        points_3D = self.DEST_PTS_WORLD
        src_pts = np.array(src_pts)
        intrinsic_matrix_file = '../config/measured/intrinsic_matrix.csv'
        distortion_matrix_file = '../config/measured/distortion_matrix.csv'
        self.loadCameraCalibration(intrinsic_matrix_file, distortion_matrix_file)
        flags = cv2.SOLVEPNP_ITERATIVE if len(points_3D) == 3 else 0
        
        success, rot_vec, trans_vec = cv2.solvePnP(points_3D, src_pts, self.intrinsic_matrix, self.distortion_matrix, flags=flags)
        
        if success:
            self.cameraCalibrated = True
            rot_matrix, _ = cv2.Rodrigues(rot_vec)
            trans_matrix = trans_vec.reshape(-1)
            self.extrinsic_matrix[0:3, 0:3] = rot_matrix
            self.extrinsic_matrix[0:3, 3] = trans_matrix
            self.extrinsic_matrix[3, 3] = 1
        else:
            self.cameraCalibrated = False
            self.extrinsic_matrix = np.eye(4)
            
        return self.cameraCalibrated
    
    def calibrateWithApriltags(self):
        """
        @brief      Calibrate with the April Tags with 4 points, and show image
        """
        # Not enough points
        if len(self.apriltag_points) < 4:
            print("INSUFFICIENT APRILTAG POINTS FOR HOMOGRAPHY")
            return False
        
        # Enough points (4 points)
        src_pts = []
        for i in [1,2,3,4]:
            src_pts.append(self.apriltag_points[i])
        src_pts = np.array(src_pts)
        self.H = cv2.findHomography(np.array(src_pts), self.DEST_PTS)[0]
        
        return True
    
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        # Deep copy of Original VideoFrame
        tag_image = self.VideoFrame.copy()
        
        # Warped Image
        if self.cameraCalibrated:
            tag_image = cv2.warpPerspective(tag_image, self.H, (tag_image.shape[1], tag_image.shape[0]))
        
        # Draw detected blocks
        tag_image = self.blockDetector(tag_image)
            
        for detection in msg.detections:
            center = detection.centre                  # centre in (x,y) pixel coordinates
            corners = []
            for i in detection.corners:
                corners.append([int(i.x), int(i.y)])   # corners of tag ((x1,y1),(x2,y2),...)
            tag_id = detection.id
            self.apriltag_points[tag_id] = [center.x, center.y]

            # Draw center
            center_color = (0, 255, 0)
            if self.H is None:
                cv2.circle(tag_image, (int(center.x), int(center.y)), 2, center_color, -1)
            else:
                center_point = np.array([[center.x], [center.y], [1]])
                center_point_homography = self.H @ center_point
                center_point_homography = center_point_homography / center_point_homography[2]
                cv2.circle(tag_image, (int(center_point_homography[0]), int(center_point_homography[1])), 2, center_color, -1)
            
            # Draw edges
            corners_array = np.array(corners, np.int32)
            corners_array = corners_array.reshape((-1, 1, 2))
            is_closed = True
            edges_color = (0, 0, 255)
            if self.H is None:
                cv2.polylines(tag_image, [corners_array], is_closed, edges_color, 2)
            else:
                corners_array = corners_array.astype(np.float32)
                corners_homography = cv2.perspectiveTransform(corners_array, self.H).astype(np.int32)
                cv2.polylines(tag_image, [corners_homography], is_closed, edges_color, 2)

            # Draw IDs
            text_color = (255, 0, 0)
            if self.H is None:
                cv2.putText(tag_image, 'ID: ' + str(tag_id), (int(center.x)+40, int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            else:
                cv2.putText(tag_image, 'ID: ' + str(tag_id), (int(center_point_homography[0])+40, int(center_point_homography[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            self.TagImageFrame = tag_image
    
    """  Pixel/ World Coordinate  """
    def getExtrinsicLegacy(self):
        """
        @brief      Get Extrinsic Matrix by manual measurement
        """
        # Rotation Matrix
        alpha = np.pi/15 + np.pi
        beta  = 0.0
        gamma = -2*np.pi/180
        Rx = np.array([[1,             0,              0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha),  np.cos(alpha)]], dtype=float)

        Ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                        [            0, 1,            0],
                        [-np.sin(beta), 0, np.cos(beta)]], dtype=float)

        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma),  np.cos(gamma), 0],
                        [            0,              0, 1]], dtype=float)
        R = Rz @ Ry @ Rx

        # Translation Vector
        T = np.array([12, 172, 1045]).reshape(-1, 3)
        
        # Extrinsic Matrix
        E = np.eye(4)
        E[0:3, 0:3] = R
        E[0:3, 3] = T
        E[3, 3] = 1
        
        return True, E 
    
    def getPixelCoordinateWithDepth(self, pt, isCalibratedView=False, isBlockDetection=False):
        """
        @brief      Given original pixel coordinate(mouse coordinate), return the warped pixel coordinate(not mouse coordinate)
        """
        pixel_coords_initial = np.array([pt.x(), pt.y(), 1])

        mouse_coords_homogenous = pixel_coords_initial

        # Unwarp to get mouse coords
        if isCalibratedView and not isBlockDetection:
            # Perform reverse homography on the cam_coords
            mouse_coords = np.linalg.inv(self.H) @ pixel_coords_initial
            mouse_coords_homogenous = mouse_coords/mouse_coords[2]
        
        z_depth = self.DepthFrameRaw[int(mouse_coords_homogenous[1])][int(mouse_coords_homogenous[0])]
        
        return int(mouse_coords_homogenous[0]), int(mouse_coords_homogenous[1]), int(z_depth), mouse_coords_homogenous
    
    def getWorldCoordinate(self, pixel_coords_homogenous, depth):
        """
        @brief      Given wawrped pixel coordinate(not mouse coordinate), return the world coordinate
        """
        # Intrinsic Matrix
        intrinsic_matrix_file = '../config/measured/intrinsic_matrix.csv'
        # intrinsic_matrix_file = '../config/factory/intrinsic_matrix.csv'
        self.loadCameraCalibration(intrinsic_matrix_file)
        K = self.intrinsic_matrix
        cam_coords_norm = np.linalg.inv(K) @ pixel_coords_homogenous
        cam_coords = cam_coords_norm * depth  # De-normalize
        cam_coords = np.hstack((cam_coords, 1))
        
        # Extrinsic Matrix
        E = np.zeros((4, 4), dtype=float)
        # AprilTag
        if self.cameraCalibrated:
            E = self.extrinsic_matrix
            # succ, E = self.getExtrinsicLegacy()
        else:
            succ, E = self.getExtrinsicLegacy()
            if not succ:
                print("ERROR WITH FINDING ROTATION, TRANSLATION VALUES")
        
        # World Coordinate
        world_coords = np.linalg.inv(E) @ cam_coords
        world_x, world_y, world_z, _ = world_coords
        
        return world_x, world_y, world_z
    
    def getWorldCoordinatesFromWarpedPixel(self, x, y, isCalibratedView=False, isBlockDetection=False):
        """
        @brief      Given world coordinate x, y, return the world coordinate z
        """
        # print("pixel coords: ", x, y)
        # Pixel coordinate -> mouse coordinate (Warped window) 
        pixel_coords_warped = np.array([x, y, 1])
        mouse_coords = np.linalg.inv(self.H) @ pixel_coords_warped
        mouse_coords = mouse_coords/mouse_coords[2]
        x, y = int(mouse_coords[0]), int(mouse_coords[1])
        # print("mouse coords: ", x, y)

        _, _, z_depth, pixel_coords_homo = self.getPixelCoordinateWithDepth(QPoint(x, y), isCalibratedView=isCalibratedView, isBlockDetection=isBlockDetection)
        world_x, world_y, world_z = self.getWorldCoordinate(pixel_coords_homo, z_depth)
        
        return world_x, world_y, world_z

    def world2DToWarpedPixel(self, x, y):
        return x+630, y+525
    
    def warpedPixelToWorld3D(self, x, y):
        return x-630, y-525, 0
    
    def getWorldDepth(self):
        """ Not in use """
        height, width = self.DepthFrameRaw.shape
        
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        d = self.DepthFrameRaw.flatten()

        # Back-projection to Camera Space (3D)
        uv1 = np.stack((u, v, np.ones_like(u)), axis=-1)
        camera_coords = d[:, np.newaxis] * (uv1 @ np.linalg.inv(self.intrinsic_matrix).T)
        
        # Homogeneous Vector
        camera_coords_homogeneous = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])
        
        # World Coordinate
        world_coords = self.extrinsic_matrix @ camera_coords_homogeneous.T
        world_coords_img = world_coords[:3, :].T.reshape(height, width, 3)
        
        return world_coords_img

    def gridToWorldCoordinate(self):
        world_coords_3d = np.array([self.warpedPixelToWorld3D(x, y) for x, y in self.grid_coords])
        world_coords_3d[:, 1] = -world_coords_3d[:, 1]
        
        return world_coords_3d

    def printError(self):
        
        if self.cameraCalibrated: 
            grid_world_3d = np.array([self.getWorldCoordinatesFromWarpedPixel(x, y, isCalibratedView=True, isBlockDetection=False) for x, y in self.grid_coords])
            grid_world_3d_ground_truth = self.gridToWorldCoordinate()
            
            # Compute errors
            error_x = grid_world_3d[:, 0] - grid_world_3d_ground_truth[:, 0]
            error_y = grid_world_3d[:, 1] - grid_world_3d_ground_truth[:, 1]
            error_z = grid_world_3d[:, 2] - grid_world_3d_ground_truth[:, 2]
            
            # Compute norms of errors
            error_vectors = np.vstack((error_x, error_y, error_z)).T
            norm_err = np.linalg.norm(error_vectors, axis=1)
            norm_err = norm_err.reshape((14, 21))
            # Plot the error norms using imshow
            im = plt.imshow(norm_err, interpolation='nearest', vmax=170)
            
            # Add a colorbar
            plt.colorbar(im, label='Error Norm')
            
            plt.title('Error Norm Visualization')
            plt.show()
            
                        
            # Plot histogram of norms
            # plt.hist(norm_err, bins=50, color='blue')
            # plt.title('Histogram of Error Norms')
            # plt.show()
            
            
class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        

class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # if self.camera.cameraCalibrated:
        #     self.camera.DepthFrameRaw = cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.H, (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()