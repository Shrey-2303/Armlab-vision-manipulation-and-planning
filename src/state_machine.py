"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import matplotlib.pyplot as plt
import kinematics as kin

DEBUG_CLOCK_RATE = 0.1

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.joint_names = ["base", "shoulder", "elbow", "wrist angle", 'wrist rotate']
        self.waypoints = [
            [[-np.pi/2,            -0.5,         -0.3,          0.0,          0.0], False],
            [[0.75*-np.pi/2,        0.5,          0.3,     -np.pi/3,      np.pi/2],  True],
            [[0.5*-np.pi/2,        -0.5,         -0.3,    np.pi / 2,          0.0], False],
            [[0.25*-np.pi/2,        0.5,          0.3,     -np.pi/3,      np.pi/2],  True],
            [[0.0,                  0.0,          0.0,          0.0,          0.0], False],
            [[0.25*np.pi/2,        -0.5,         -0.3,          0.0,      np.pi/2],  True],
            [[0.5*np.pi/2,          0.5,          0.3,     -np.pi/3,          0.0], False],
            [[0.75*np.pi/2,        -0.5,         -0.3,          0.0,      np.pi/2],  True],
            [[np.pi/2,              0.5,          0.3,     -np.pi/3,          0.0], False],
            [[0.0,                  0.0,          0.0,          0.0,          0.0],  True]]
        
        self.taught_waypoints = []
        self.gripper_status = False
        self.pp_state = False

        # diagnostics use
        self.joint_angles = []
        self.eepose = []
        self.record_joint_angles = False
        self.publish_joint_angles = False
        
        self.rxarm.set_moving_time(1.2)
        self.rxarm.set_accel_time(0.25)

    def set_gripper(self, gripper_state):
        self.gripper_status = gripper_state
        self.rxarm.enable_gripper_torque()
        print("gripper toggled", self.gripper_status)

        if self.gripper_status:
            self.rxarm.gripper.grasp()
        else:
            self.rxarm.gripper.release()    

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state
    
    def run_diagnositcs(self):
        if self.record_joint_angles:
            self.joint_angles.append(self.rxarm.get_positions())
            self.eepose.append(self.rxarm.get_ee_pose())
            print("recording angles")

        if self.publish_joint_angles:
            self.record_joint_angles = False
            self.publish_joint_angles = False

            ja = np.array(self.joint_angles)
            
            # t=np.arange(0, DEBUG_CLOCK_RATE, DEBUG_CLOCK_RATE*len(self.joint_angles))
            for i in range(5):
                plt.plot(ja[:,i], label=self.joint_names[i])
            
            plt.legend(["base","shoulder","elbow","wrist angle","wrist rotation"])
            plt.xlabel("Timesteps")
            plt.ylabel("rad")
            plt.suptitle("joint angles over time in rad")
            plt.savefig("/home/student_pm/armlab-f-23-s-8/data/publish_joint_angles.jpg")
            
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            jb = np.array(self.eepose)
            x = jb[:,0]
            y = jb[:,1]
            z = jb[:,2]
            print(x)
            print(jb)
            
            ax.plot(x, y, z)
            ax.set_title("trajectory of end effector")
            ax.figure.savefig("/home/student_pm/armlab-f-23-s-8/data/ee.jpg")


            print("PUBLISH JOINT ANGLES")
            self.joint_angles = []

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.record_joint_angles = True #start recording joint angles
            self.execute(self.waypoints)
            self.publish_joint_angles = True
        
        if self.next_state == "execute taught":
            self.record_joint_angles = True #start recording joint angles
            self.execute(self.taught_waypoints)
            self.publish_joint_angles = True
        
        if self.next_state == "prep_calibrate":
            self.prep_calibrate()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record":
            self.record()
        
        if self.next_state == "pick and click":
            self.pick_click()

        if self.next_state == "clear_taught_waypoints":
            self.clear_taught_waypoints()
    
    """  Basic Modes  """
    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()
    
    """  Record  """
    def record(self):
        self.status_message = "State: Record - The record button to teach the robot points to move to"
        self.current_state = "record"
        
    def toggle_record(self):
        print("TOGGLE RECORD MODE: ")
        if self.current_state == "idle":
            # switch to record state
            self.wypts = []
            self.next_state = "record"
            self.rxarm.disable_torque()
            print(self.next_state)

        elif self.current_state == "record":
            self.next_state = "idle"
            self.rxarm.enable_torque()
            print(self.next_state)
            # switch to idle state
    
    def record_point(self):
        # self.taught_waypoints = []
        if self.current_state != "record":
            self.status_message  = "SWITCH STATE TO RECORD STATE TO RECORD A POINT"
            print(self.status_message)
            return
        
        self.taught_waypoints.append([list(self.rxarm.get_positions()), self.gripper_status])
        print("CURRENT TAUGHT WAYPOINTS", self.taught_waypoints)
        return

    def clear_taught_waypoints(self):
        self.taught_waypoints = []

    def execute(self, wypts):
        """!
        @brief      Go through all waypoints
        """
        if self.current_state != "idle":
            self.status_message  = "State Switch Failure"
            return
        self.status_message = "State: Execute - Executing motion plan"

        self.current_state = "execute"
        for position, gripper in wypts:
            # Gripper
            if gripper:
                self.rxarm.gripper.grasp()
            else:
                self.rxarm.gripper.release()
            
            time.sleep(0.25)
            self.rxarm.set_positions(position)
            self.check_motion(position)

            
            
        self.next_state = "idle"

    def clear_taught_waypoints(self):
        self.taught_waypoints = []

    """  Click and Pick  """
    def pick_click(self):
        self.status_message = "State: Click anywhere on the video for the robot to move to"
        self.current_state = "pick and click"
        
    def toggle_pick_click(self):
        print("state before checking: ",self.current_state)
        if self.current_state == "idle":
            self.next_state = "pick and click"
            print(self.next_state)
        
        elif self.current_state == "pick and click":
            self.next_state = "idle"
            print(self.next_state)
    
    def pick_click_IK(self, pose):
        """!
        @ brief      pick click function
        """
        # Pick
        if self.gripper_status == False:
            self.addToWaypoints(pose, grab=True, angle=0)
        # Place
        else: 
            self.addToWaypoints(pose, grab=False, angle=0)
        
        self.execute(self.taught_waypoints)
            
        
 
    def check_motion(self, angles):
        x = time.time()
        while(True):
            time.sleep(0.1)
            if np.all(abs(np.array([list(self.rxarm.get_positions())]) - np.array(list(angles))) < 0.08) == True or time.time() - x > 2: 
                break
        
    """  Calibration  """
    def run_calibrate(self):
        if self.current_state == "prep_calibrate":
            self.next_state = "calibrate"
        elif self.current_state!= "calibrate":
            self.next_state = "prep_calibrate"

    def prep_calibrate(self):
        pt5 = self.camera.DEST_PTS_WORLD[4]
        pt6 = self.camera.DEST_PTS_WORLD[5]
        pt7 = self.camera.DEST_PTS_WORLD[6]
        self.status_message = (
            "State: Calibration - PLEASE PLACE "
            f"ID 5 at ({int(pt5[0])}, {int(pt5[1])}), "
            f"ID 6 at ({int(pt6[0])}, {int(pt6[1])}), and "
            f"ID 7 at ({int(pt7[0])}, {int(pt7[1])}). "
        )
        self.next_state = "prep_calibrate"
        self.current_state = "prep_calibrate"
        
    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """

        """TODO Perform camera calibration routine here"""
        self.current_state = "calibrate"
        self.next_state = "idle"
        
        if self.camera.findExtrinsicWithAprilTag():
            pass
        else:
            # fail
            self.status_message = "Calibration - SolvePnP Calibration Failed"
            return
            
        if self.camera.calibrateWithApriltags():
            self.status_message = "Calibration - Completed Calibration"
        else:
            self.status_message = "Calibration - Homography transform Failed"
            return
        
        time.sleep(1)

    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(0.5)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        time.sleep(3)
        self.rxarm.poses_at_init = list(self.rxarm.get_positions())
        print("POSES AT INIT: ", self.rxarm.poses_at_init)
        self.rxarm.set_moving_time(1.2)
        self.rxarm.set_accel_time(0.25)
        self.next_state = "idle"
    
    """  Challenge definition  """
    def block_size_sort(self, task=''):
        block_detections = self.camera.block_detections
        if task == "task_3" or task=='task_4':
            large_blocks = {}
            small_blocks = {}
        else:
            small_blocks = []
            large_blocks = []

        all_blocks = []

        # Categorize blocks and store existing block centroids
        for color, blocks in block_detections.items():
            for block in blocks:
                is_large = block['is_large']
                x, y = block['centroid']
                rot = block['rot']
                
                isBlockDetection = True
                # pose = [x, y, z, rot, is_large, color]
                pose = list(self.camera.getWorldCoordinatesFromWarpedPixel(x, y, self.camera.cameraCalibrated, isBlockDetection))
                pose.extend([rot, is_large, color])
                # check if block center distance is too close to any previous blocks
                # if previous block in tolerance check which z value is higher in pose
                # replace if new block is higher in pose, otherwise ignore current detection
                print("all blocks pose:: ", pose)
                for i, det in enumerate(all_blocks):
                    if np.linalg.norm(np.array(pose[:2])-np.array(det[:2])) < 40:
                        if pose[2] > det[2]:
                            all_blocks[i] = pose
                        break
                else:
                    all_blocks.append(pose)

                # print("World coordinate: ", pose)
        for pose in all_blocks:
            is_large = pose[4]
            xyz_rot = pose[:4]
            xyz_rot_size = pose[:5]
            # task1, 2 -> pose = [x, y, z, rot, is_large]
            if task == 'task_1':
                if is_large and pose[1] > -10:
                    large_blocks.append(xyz_rot)
                elif is_large == False and pose[1] > -10:
                    small_blocks.append(xyz_rot)
            elif task == 'task_2':
                if is_large and pose[1] > 20:
                    large_blocks.append(xyz_rot)
                elif is_large == False and pose[1] > 20:
                    small_blocks.append(xyz_rot)
            elif task == 'task_3':
                color = pose[5]
                if is_large:
                    large_blocks[color] = pose[:4]
                else:
                    small_blocks[color] = pose[:4]    
            else:
                large_blocks.append(pose[:4])
                small_blocks.append(pose[:4])

        if not task == 'task_3':
            for i, block in enumerate(all_blocks):
                all_blocks[i] = all_blocks[i][:]
                    
        return all_blocks, small_blocks, large_blocks

    def block_in_range(self, dest, blocks, THRESH):
        #checks all the blocks and returns the first block that is within distance threshold
        # returns None if None exist
        print (dest,blocks)
        for block in blocks:
            if np.linalg.norm(np.array(block[0:2])-np.array(dest[0:2]))<THRESH:
                return block
            
        return None
        
    def addToWaypoints(self, location:list, grab:bool, angle:int, is_large = False):
        """
        @brief: updates taught waypoints given a target location for the arm to pick up a block or put down a block
        """
        # ABOVE: move to a place to prepare the grabbing operation
        angle = (90-angle)/180*np.pi

        # ABOVE: move to a place to prepare the grabbing operation
        above_height = 120
        angles, form = kin.IK_geometric([location[0],location[1],location[2]+above_height], self.rxarm.poses_at_init, yaw = angle)
        self.taught_waypoints.append([angles, not grab])
        
        # Setting offset for pick and place, large and small
        offset = 0
        # Small
        if is_large:
            if grab:
                offset = 10
            else:
                offset = 50
        # Large
        else:
            if grab:
                offset = 10
            else:
                offset = 50
        
        location[2] = location[2] + offset
                    
        # GRAB: move in to grab the block
        angles, form = kin.IK_geometric([location[0],location[1],location[2]], self.rxarm.poses_at_init, yaw = angle, normal_form = form)
        self.taught_waypoints.append([angles, not grab])
        #       commit to grabbing or releasing
        self.taught_waypoints.append([angles, grab])

        # ABOVE INTREMIDIATE: move out of that location
        intermidiate_height = 60
        angles,form = kin.IK_geometric([location[0],location[1],location[2]+intermidiate_height], self.rxarm.poses_at_init, yaw = angle, normal_form = form)
        self.taught_waypoints.append([angles, grab])

        # ABOVE: move out of that location
        angles,form = kin.IK_geometric([location[0],location[1],location[2]+above_height], self.rxarm.poses_at_init, yaw = angle)
        self.taught_waypoints.append([angles, grab])

    def setPicturePose(self):
        PICTURE_POSE, _ = kin.IK_geometric([0, 50, 70], self.rxarm.poses_at_init, 0)
        self.rxarm.set_positions(PICTURE_POSE)
        self.check_motion(PICTURE_POSE)
        
    """ Task 1 """
    def task1(self):
        start = time.time()
        # Task 1 have to calibrate first
        if not self.camera.cameraCalibrated:
            print("ERROR: attempting to execute a task with no calibration data")
            self.status_message = "ERROR: attempting to execute a task with no calibration data"
            return
        
        is_finsihed = False
        self.initialize_rxarm()

        while not is_finsihed:
            # Find the blocks that need to be picked up
            all_blocks, small_blocks, large_blocks = self.block_size_sort('task_1')

            # Check the blocks are in the correct location
            if len(small_blocks) == 0 and len(large_blocks) == 0:
                is_finsihed = True
            
            # Empty Taught Waypoints
            self.clear_taught_waypoints()
            
            large_place_location, small_place_location = self.camera.find_place_location(all_blocks, 'task_1')
            print(f"Large place: {len(large_place_location)}")
            print(f"Small place: {len(small_place_location)}\n")

            large_count = small_count = 0
            for block in small_blocks:
                block_pose = block[:3]
                block_rot = block[3]
                # Pick
                self.addToWaypoints(block_pose, grab = True, angle=block_rot, is_large = False)
                # Place
                self.addToWaypoints(small_place_location[small_count], grab=False, angle=90, is_large = False)
                small_count += 1
                
            for block in large_blocks:
                block_pose = block[:3]
                block_rot = block[3]
                # Pick
                self.addToWaypoints(block_pose, grab = True, angle=block_rot, is_large = True)
                # Place
                self.addToWaypoints(large_place_location[large_count], grab=False, angle=90, is_large = True)
                large_count += 1            
            
            self.execute(self.taught_waypoints)
        
        self.rxarm.sleep()
        print("============   Task 1 Complete   ============")
        print(f"time: {int(time.time() - start)} sec")
        
    """ Task 2 """
    def stack_place_block(self, place_location, is_large=False):
        
        # check all the blocks
        if is_large:
            all_blocks, _, blocks = self.block_size_sort('task_2')
            current_stack = self.block_in_range(place_location, all_blocks, 50)
            if current_stack != None:
                place_location[2] = current_stack[2]
            else:
                place_location[2] = 0 
            print("large blocks left: ", len(blocks))
        else:
            all_blocks, blocks, _ = self.block_size_sort('task_2')
            current_stack = self.block_in_range(place_location, all_blocks, 50)
            if current_stack != None:
                place_location[2] = current_stack[2]
            else:
                place_location[2] = 0 
            print("small blocks left: ", len(blocks))
        is_complete = False
        
        # check wether the blocks are finish stacking
        if len(blocks) == 0:
            is_complete = True
            return is_complete
        
        # Pick block location - process one block at a time
        block_pose = blocks[0][:3]
        block_rot = blocks[0][3]
        self.addToWaypoints(block_pose, grab = True, angle=block_rot, is_large=is_large)

        self.addToWaypoints(place_location, grab=False, angle=90)
        
        print("  current stack: ", current_stack)
        print("  Pick location: ", block_pose)
        print("  Place location: ", place_location)
        
        return is_complete
   
    def task2(self):
        start = time.time()
        place_locations = np.array([
            [-250, -25, 0], 
            [250, -25, 0]
        ])
        
        small_complete = large_complete = False
        self.initialize_rxarm()
        
        # Pick the small blocks
        while not small_complete:            
            # Clear the Waypoints
            self.clear_taught_waypoints()

            # Add Waypoints to large blocks and small blocks
            small_complete = self.stack_place_block(place_locations[1])
            
            self.execute(self.taught_waypoints)
            
            self.setPicturePose()
        print("====   Small Blocks Complete   ====")
        
        while not large_complete:            
            # Clear the Waypoints
            self.clear_taught_waypoints()

            # Add Waypoints to large blocks and small blocks
            large_complete = self.stack_place_block(place_locations[0], is_large=True)
            
            self.execute(self.taught_waypoints)
            
            self.setPicturePose()
        print("====   Large Blocks Complete   ====")
        
        self.rxarm.sleep()
        print("============   Task 2 Complete   ============")
        print(f"time: {int(time.time() - start)} sec")
        
    def remove_distractor(self, obstical):
        print("BLOCKED BY:", obstical)
        if obstical[5] == None: #if no color data is associated with this obstical
            #remove
            #pick up obstical
            angle = 180- obstical[3]
            self.addToWaypoints(obstical[:3], grab=True, angle=obstical[3])
            
            #remove
            angles, form = kin.IK_geometric([150,-150,150], self.rxarm.poses_at_init)
            self.taught_waypoints.append([angles, True])
            #       commit to grabbing or releasing
            self.taught_waypoints.append([angles, False])
    
    """ Task 3 """
    def task3(self):
        if not self.camera.cameraCalibrated:
            print("ERROR: attempting to execute a task with no calibration data")
            self.status_message = "ERROR: attempting to execute a task with no calibration data"
            return
        
        PICTURE_POSE, form = kin.IK_geometric([0, 50, 70], self.rxarm.poses_at_init, 0)
        
        large_line = {"Red": [370,-50,0], "Orange": [320,-50,0], "Yellow":[270,-50,0], "Green":[220,-50,0], "Blue":[170,-50,0], "Violet":[120,-50,0]}
        small_line = {"Red": [-370,-50,0], "Orange": [-320,-50,0], "Yellow":[-270,-50,0], "Green":[-220,-50,0], "Blue":[-170,-50,0], "Violet":[-120,-50,0]}
        
        # compute large line
        DIST_THRESH = 40
        SMALL_DIST_THRESH = 25
        line_complete = False
        self.initialize_rxarm()
        while not line_complete:
            # go to picture position
            self.rxarm.set_positions(PICTURE_POSE)
            self.clear_taught_waypoints()
            self.check_motion(PICTURE_POSE)
            
            #gets block detections
            all_blocks, small_blocks_color, large_blocks_color = self.block_size_sort("task_3")
            
            #append obstacle blocks to all blocks:
            for det in self.camera.irregular_detections:
                pose = list(self.camera.getWorldCoordinatesFromWarpedPixel(det["centroid"][0], det['centroid'][1], self.camera.cameraCalibrated, True))
                pose.append(det['rot'])
                pose.append(None) #is large
                pose.append(None) #color
                all_blocks.append(pose)

            for color in large_line:
                print("large ", color)
                obstacle = self.block_in_range(large_line[color], all_blocks, DIST_THRESH)
                if obstacle == None: #if nothing blocks the way
                    if color in large_blocks_color.keys():
                        block = large_blocks_color[color]
                        print("moving: ", color, " to: ", large_line[color])
                        #pick and place to destination
                        self.addToWaypoints(block[:3], grab=True, angle=block[3], is_large = True)
                        self.addToWaypoints(large_line[color], grab=False, angle=90, is_large = True)
                else:
                    self.remove_distractor(obstacle)
            
            for color in small_line:
                print("small ", color)
                obstacle = self.block_in_range(small_line[color], all_blocks, SMALL_DIST_THRESH)
                if obstacle == None:#if nothing blocks the way
                    if color in small_blocks_color.keys():
                        block = small_blocks_color[color]
                        print("moving: ", color, " to: ", small_line[color])
                        #pick and place to destination
                        self.addToWaypoints(block[:3], grab=True, angle=block[3], is_large = False)
                        self.addToWaypoints(small_line[color], grab=False, angle=90, is_large = False)
                else:
                    self.remove_distractor(obstacle)

            
            if len(self.taught_waypoints) == 0:
                line_complete = True
            else:
                self.execute(self.taught_waypoints)
                print(self.taught_waypoints)
            
            
        self.rxarm.sleep()
        
        #check if large line is occupied
        #rotate through all the colors, check for large blocks and save
        #place in location check again excluding for region in the line

    """ Task 4 """
    def task4(self):
        print(self.camera.extrinsic_matrix)
        is_on_surface = False
        self.initialize_rxarm()

        # Make all the blocks not stacked
        while not is_on_surface:
            all_blocks, small_blocks, large_blocks = self.block_size_sort('xyz_rot_size')

            # Check the blocks are on the surface
            if len(all_blocks) == 12 and len(small_blocks) == 6 and len(large_blocks) == 6:
                is_on_surface = True

            # Empty Taught Waypoints
            self.clear_taught_waypoints()

            # Find only the highest block
            highest_block = 0
            highest_pose = None
            highest_rot = None
            highest_is_large = None
            for block in all_blocks:
                if highest_block < block[2]:
                    highest_block = block[2]
                    highest_pose = block[:3]
                    highest_rot = block[3]
                    highest_is_large = block[4]

            # Add Waypoints to highest block
            self.addToWaypoints(highest_pose, grab = True, angle=highest_rot, is_large=highest_is_large)

            # Place the blocks
            large_place, small_place = self.camera.find_place_location(all_blocks, 'task_4')
            if highest_is_large:
                self.addToWaypoints(large_place[0], grab=False, angle=90, is_large=True)
            else:
                self.addToWaypoints(small_place[0], grab=False, angle=90, is_large=False)

            self.execute(self.taught_waypoints)

        # Start stacking the colors
        stack_locations = np.array([
            [-150, -50, 0], 
            [150, -50, 0]
        ])

        colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Violet"]
        large_stack = {color: stack_locations[0] for color in colors}
        small_stack = {color: stack_locations[1] for color in colors}
        
        color_stacked = False
        while not color_stacked:

            for color in large_stack:
                # obstacle = self.block_in_range(large_stack[color], all_blocks, 30)
                all_blocks, _, large_blocks_color = self.block_size_sort('task_3')
                large_count = len(large_blocks_color)

                self.addToWaypoints(large_blocks_color[color], grab=True, angle=large_blocks_color[color][3], is_large=True)

                self.stack_place_block(large_blocks_color, stack_locations[1], max_stack=6, large_count=large_count, is_large=True)

                self.execute(self.taught_waypoints)

            for color in small_stack:
                # obstacle = self.block_in_range(large_stack[color], all_blocks, 30)
                all_blocks, small_blocks_color, _ = self.block_size_sort('task_3')
                small_count = len(small_blocks_color)

                self.addToWaypoints(small_blocks_color[color], grab=True, angle=small_blocks_color[color][3], is_large=True)

                self.stack_place_block(small_blocks_color, stack_locations[1], max_stack=6, small_count=small_count, is_large=True)

                self.execute(self.taught_waypoints)
            
        
        
        

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)

class StateMachineDebugThread(QThread):
    """!
        @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine
    
    
    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run_diagnositcs()
            time.sleep(DEBUG_CLOCK_RATE)