import numpy as np
import matplotlib.pyplot as plt

#pixel coordinate of fire in image
p = [-12,-24]

#focal length
f = 0.06

#uav position vector
uav_coord = np.array([[4],
                     [5],
                     [16],
                     [1]])

#translation from uav center to body
body_trans = [3, -2, 4]
#rotation of uav body(roll, pitch, yaw) in degrees
body_rot = [15, -2, 45]

#translation from body to gimbal
gimbal_trans = [1,2,3]

#rotation of gimbal relative to body frame
gimbal_rot = [0,23,21]

#translation from gimbal to camera center 
cam_trans = [0,2,0]

#rotaion of camera frame relative to gimbal frame
cam_rot = [90,0,90]

#rotation of image relative to camera frame
image_rot = [0, 0, 0]


#transformation function that takes translation, and rotation 
def transform(vector, trans, rotation):
  phi = np.deg2rad(rotation[2])
  theta = np.deg2rad(rotation[1])
  psi = np.deg2rad(rotation[0])

  rotz = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                   [np.sin(phi), np.cos(phi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

  roty = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta), 0, np.cos(theta), 0],
                   [0, 0, 0, 1]])

  rotx = np.array([[1, 0, 0, 0],
                   [0, np.cos(psi), -np.sin(psi), 0],
                   [0, np.sin(psi), np.cos(psi), 0],
                   [0 , 0, 0, 1]])

  translation = np.array([[1, 0, 0, trans[0]],
                         [0, 1, 0, trans[1]],
                         [0, 0, 1, trans[2]],
                         [0, 0, 0, 1]])




  return np.matmul(np.matmul(translation, np.matmul(np.matmul(rotz, roty), rotx)), vector)

#function that calculates position of object relative to camera based on pixel location
def object_to_cam_transformation(fx, fy, c, p, s, f):
  p_vec = np.array([[p[0]],
                   [p[1]],
                   [1]])

  cam_mat = np.array([[fx, s, 0],
                      [0, fy, 0],
                      [0, 0, 1]])
  
  
 #print(cam_mat)

  obj_vec = np.matmul(np.linalg.inv(cam_mat), p_vec)


  x = f * obj_vec[0][0]
  y = f * obj_vec[1][0]

  return [x, y, f]


  #depth = c[2][0] / (obj_vec)


def plane_vec_intersection(p,v):

    """
    Computes the intersection of a vector with the plane z=0.
    
    Parameters:
        p (tuple): A point on the line, (x0, y0, z0)
        v (tuple): Direction vector, (vx, vy, vz)
        
    Returns:
        tuple: Intersection point (x, y, z) or None if no intersection.
    """
    x0, y0, z0 = p
    vx, vy, vz = v

    if vz == 0:
        # The line is parallel to the plane and does not intersect
        return None

    t = -z0 / vz
    x = x0 + t * vx
    y = y0 + t * vy
    z = 0  # By definition of the plane

    return (x, y, z)



def calculate_fire_pos():
    uav_body = transform(uav_coord, body_trans, body_rot)

    uav_gimbal = transform(uav_body, gimbal_trans, gimbal_rot)
    uav_cam = transform(uav_gimbal, cam_trans, cam_rot)
   
    cam_to_obj_vec = object_to_cam_transformation(0.05,0.03, uav_cam, p, 0, f)
    print(cam_to_obj_vec)

    image_point_vec = transform(uav_cam, cam_to_obj_vec, image_rot)

    print(uav_cam)
    print(image_point_vec)

    fire_direction_vec = uav_cam - image_point_vec
    print(fire_direction_vec)

    cam_point = (uav_cam[0][0], uav_cam[1][0], uav_cam[2][0])
    fire_direction_tuple = (fire_direction_vec[0][0], fire_direction_vec[0][0], fire_direction_vec[0][0])

    print(plane_vec_intersection(cam_point, fire_direction_tuple))

calculate_fire_pos()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#ax.scatter(uav_coord)


