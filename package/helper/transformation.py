import math 
import numpy as np

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def p2t(p):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    R = np.eye(3,3)
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R    

def rpy2r(rpy_rad):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def rpy2r_order(r0, order=[0,1,2]):
    """ 
        roll,pitch,yaw in radian to R with ordering
    """
    c1 = np.math.cos(r0[0]); c2 = np.math.cos(r0[1]); c3 = np.math.cos(r0[2])
    s1 = np.math.sin(r0[0]); s2 = np.math.sin(r0[1]); s3 = np.math.sin(r0[2])
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)
    return a

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2quat(R):
    """ 
        Convert Rotation Matrix to Quaternion.  See rotation.py for notes 
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def r2sixd(R):
    r1 = R[:,0]
    r2 = R[:,1]
    return np.hstack((r1,r2))

def sixd2r(d6):
    r1 = d6[:3]
    r2 = d6[3:]
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.stack((r1, r2, r3), axis=1)
    return R

def skew(x):
    """ 
        Get a skew-symmetric matrix
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        Compute the rotation matrix from an angular velocity vector
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] norm of a should be 1.0 not [%.2e]."%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R

def get_rotation_matrix_from_two_points(p_fr,p_to):
    p_a  = np.copy(np.array([0,0,1]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R

def R_yuzf2zuxf(R):
    """
        Convert R of (Y-up z-front) to (Z-up x-front)
    """
    R_offset = rpy2r(np.radians([-90,0,-90]))
    return R_offset@R

def T_yuzf2zuxf(T):
    """
        Convert R of (Y-up z-front) to (Z-up x-front)
    """
    p,R = t2pr(T)
    T = pr2t(p=p,R=R_yuzf2zuxf(R))
    return T

def rpy_deg2r(rpy_deg):
    """
        Roll Pitch Yaw in Degree to Rotation Matrix
    """
    return rpy2r(np.deg2rad(rpy_deg))