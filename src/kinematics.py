"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params,links):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    num_links = links
    #num_links = len(dh_params)
    T_0_i = np.identity(4)  

    for i in range(num_links):
        theta, d, a, alpha = dh_params[i]
        T_i_minus_1_i = get_transform_from_dh(theta, d, a, alpha)
        T_0_i = np.dot(T_0_i, T_i_minus_1_i)  

    return T_0_i

def DH_matrix(t1=0, t2=0, t3=0, t4=0, t5=0):
    dh_matrix = np.array([
                            [t1+np.pi/2, 103.91, 0, -np.pi/2],
                            #[t2+np.pi/2, 0, -200, 0],
                            #[-np.pi/2, 0, 50, 0]
                            [t2-1.3258176637, 0, np.sqrt(200**2 + 50**2), 0],
                            [t3+1.3258176637, 0, 200, 0],
                            [t4-np.pi/2, 0, 0, -1*np.pi/2],
                            [t5+np.pi, 174.15, 0, 0]
                        ])
    return dh_matrix



def get_transform_from_dh(theta, d, a, alpha):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians
    @return     The 4x4 transformation matrix.
    """
    transformation_matrix = np.array([
        [np.cos(theta), -1*np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -1*np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    # print(theta, d, a, alpha,"\n")
    # print(transformation_matrix,"\n\n\n")
    return transformation_matrix


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    # Rotation about Z-axis (yaw)
    yaw = np.arctan2(T[1, 0], T[0, 0])
    # Rotation about Y-axis (pitch)
    pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))
    # Rotation about X-axis (roll)
    roll = np.arctan2(T[2, 1], T[2, 2])


    # Rotation about the Z-axis (ψ)
    psi = yaw
    # Rotation about the X-axis (θ)
    theta = np.arctan2(np.sqrt(T[0, 2]**2 + T[1, 2]**2), T[2, 2])
    # Rotation about the Z-axis again (ϕ)
    phi = np.arctan2(T[2, 1], -T[2, 0])
    #return psi, theta, phi

    return np.array([roll, pitch, yaw])


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    angles = get_euler_angles_from_T(T)
    positions = T[:3,3:4].flatten()
    pos = list(np.append(positions,angles))
    pos_rounded = list(map(lambda x: round(x, ndigits=2), pos))
    return pos_rounded


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint anIK_geometric
    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass

def manual_rotation(roll, pitch, yaw):

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_combined = R_yaw @ R_pitch @ R_roll

    return R_combined

def normal_form_calculation(pose, yaw):
    """!
    @brief     Compute the normal form based on the given pose and yaw angle
    """
    roll = 0
    pitch = np.pi

    end_effector = np.array([pose[0],pose[1],pose[2]])   # pose[0,1,2] = x,y,z   #transformation_matrix_FK[:3,3]
    R = manual_rotation(roll, pitch, yaw)
    
    d5 = 174.15  
    z_unit_vector = np.array([0, 0, 1])
    O4_position = end_effector.T - d5*(R @ z_unit_vector.T)
    
    r = np.sqrt(O4_position[0]**2 + O4_position[1]**2)
    s = O4_position[2] - 103.91
    
    return r, s, O4_position, R

def determine_formation(pose, r, s, O4, R, is_normal_form=None):
    # Initial check
    if is_normal_form == None:
        q3_angle_succ = -1 <= (r**2 + s**2 - 82500)/(2*200*np.sqrt(42500)) <= 1
    # Given specific form
    elif is_normal_form:
        q3_angle_succ =True
    else:
        q3_angle_succ =False
        
    # Form 1
    if q3_angle_succ:
        q3 = np.arccos((r**2 + s**2 - 82500)/(2*200*np.sqrt(42500))) - np.arctan2(4,1) 
        q2 = np.pi/2 - np.arctan2(1,4) - np.arctan2(s,r) - np.arcsin(((200*np.sin(np.arctan2(4,1) + q3))/np.sqrt(r**2 + s**2)))
        q1 = np.arctan2(-O4[0],O4[1])
        dh_matrix_IK = DH_matrix(q1,q2,q3)
        T_matrix = FK_dh(dh_matrix_IK,3)
        R_3_5 = T_matrix.T[:3,:3] @ R
        q4 = np.arctan2(R_3_5[1,2],R_3_5[0,2])- (6*np.pi/180)
        q5 = np.arctan2(-R_3_5[2,1],R_3_5[2,0])
    
    # Form 2
    else: 
        roll = np.arctan2(-pose[0],pose[1])
        y = (np.sqrt(pose[0]**2 + pose[1]**2) - 170.15) * np.sin(roll)
        x = (np.sqrt(pose[0]**2 + pose[1]**2) - 170.15) * np.cos(roll)
        O4 = [x,y,pose[2]]
        r = np.sqrt(O4[0]**2 + O4[1]**2)
        s = O4[2] - 103.91
        try:
            q3 = np.arccos((r**2 + s**2 - 82500)/(2*200*np.sqrt(42500))) - np.arctan2(4,1)
            q2 = np.pi/2 - np.arctan2(1,4) - np.arctan2(s,r) - np.arcsin(((200*np.sin(np.arctan2(4,1) + q3))/np.sqrt(r**2 + s**2)))
        except:
            print("IK ERROR, CANNOT REACH POSITION:", pose)
            return 0, 0, 0, 0, 0

        q1 = roll
        q4 = - (q2 + q3 + (np.arctan2(1,4)/2) + (2 * np.pi/180))
        q5 = 0
        
    return q1, q2, q3, q4, q5, q3_angle_succ

def IK_geometric(pose, correction_anlges,yaw = np.pi/2, normal_form = None):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    r, s, O4, R = normal_form_calculation(pose, yaw)
    
    # Return the joint configurations, and determine the formation of the manipulator
    q1, q2, q3, q4, q5, is_normal_form = determine_formation(pose, r, s, O4, R, normal_form)
    
    q1 = q1 + correction_anlges[0]
    q2 = q2 + correction_anlges[1]
    q3 = q3 + correction_anlges[2]
    q4 = q4 + correction_anlges[3]
    q5 = q5 + correction_anlges[4]
    q = [q1, q2, q3, q4, q5]

    return q, is_normal_form

