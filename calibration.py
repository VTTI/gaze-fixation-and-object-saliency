# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:32:48 2022

@author: Bhagat Hirva

"""
import numpy as np
import math as m

def Rx(theta):
    
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_rotation_matrix(roll, pitch, yaw):
    """ Convert Euler angles to a rotation matrix"""

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack([
        [ones,  zeros,     zeros],
        [zeros, cos_roll, -sin_roll],
        [zeros, sin_roll,  cos_roll]])

    r_pitch = np.stack([
        [ cos_pitch, zeros, sin_pitch],
        [ zeros,     ones,  zeros],
        [-sin_pitch, zeros, cos_pitch]])

    r_yaw = np.stack([
        [cos_yaw, -sin_yaw, zeros],
        [sin_yaw,  cos_yaw, zeros],
        [zeros,    zeros,   ones]])

    pose = np.einsum('ijhw,jkhw,klhw->ilhw',r_yaw,r_pitch,r_roll)
    pose = pose.transpose(2,3,0,1)
    return pose 

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = m.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = m.atan2(R[2,1] , R[2,2])
        y = m.atan2(-R[2,0], sy)
        z = m.atan2(R[1,0], R[0,0])
    else :
        x = m.atan2(-R[1,2], R[1,1])
        y = m.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def rot_vec(R,p,y):
    #R = Rz(p) * Ry(y) * Rz(0)
    vec=np.array([p,y,0],dtype='float32')
    rot_vec=R.dot(vec)
    #anglediff = (rot_vec[0] - rot_vec[1] + 180 + 360) % 360 - 180
    return rot_vec


