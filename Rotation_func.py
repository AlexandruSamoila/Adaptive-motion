from scipy.spatial.transform import Rotation as R
import numpy as np

def angle_axis_to_euler_zyz(angle, axis):

    # Using Rodrigues' rotation formula
    # Convert first to rotation matrix
    rot_mat = np.zeros((3,3))
    
    rot_mat[0][0] = np.cos(angle) + axis[0] * axis[0] * (1 - np.cos(angle))
    rot_mat[0][1] = axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle)
    rot_mat[0][2] = axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)
    rot_mat[1][0] = axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle)
    rot_mat[1][1] = np.cos(angle) + axis[1] * axis[1] * (1 - np.cos(angle))
    rot_mat[1][2] = axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)
    rot_mat[2][0] = axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle)
    rot_mat[2][1] = axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle)
    rot_mat[2][2] = np.cos(angle) + axis[2] * axis[2] * (1 - np.cos(angle))

    r = R.from_matrix(rot_mat)

    eul_zyz = r.as_euler('zyz', degrees=False)

    return eul_zyz

def quat_to_euler_zyz(quaternion):

    r = R.from_quat(quaternion)

    eul_zyz = r.as_euler('zyz', degrees=False)

    return eul_zyz


if __name__ == '__main__':

    angle_axis_to_euler_zyz(1,1)