import os,time
import torch
import numpy as np
import cvxpy as cp
import shapely as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from shapely import Polygon,LineString,Point # handle polygons

"""
sys.path.append('../../package/helper/')
 => this should be called before calling 'from mujoco_parser import MuJoCoParserClass' 
"""
from transformation import (
    t2p,
)

def d2r(deg):
    return np.radians(deg)

def r2d(rad):
    return np.degrees(rad)

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

def get_colors(n_color=10,cmap_name='gist_rainbow',alpha=1.0):
    """ 
        Get diverse colors
    """
    colors = [plt.get_cmap(cmap_name)(idx) for idx in np.linspace(0,1,n_color)]
    for idx in range(n_color):
        color = colors[idx]
        colors[idx] = color
    return colors

def compute_view_params(camera_pos,target_pos,up_vector=np.array([0,0,1])):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def meters2xyz(depth_img,cam_matrix):
    """
        Scaled depth image to pointcloud
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def get_idxs(list_query,list_domain):
    """ 
        Get corresponding indices of either two lists or ndarrays
    """
    if isinstance(list_query,list) and isinstance(list_domain,list):
        idxs = [list_query.index(item) for item in list_domain if item in list_query]
    else:
        print("[get_idxs] inputs should be 'List's.")
    return idxs

def get_idxs_closest_ndarray(ndarray_query,ndarray_domain):
    return [np.argmin(np.abs(ndarray_query-x)) for x in ndarray_domain]

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def finite_difference_matrix(n, dt, order):
    """
    n: number of points
    dt: time interval
    order: (1=velocity, 2=acceleration, 3=jerk)
    """ 
    # Order
    if order == 1:  # velocity
        coeffs = np.array([-1, 1])
    elif order == 2:  # acceleration
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # jerk
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # Fill-in matrix
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c

    # (optional) Handling boundary conditions with backward differences
    if order == 1:  # velocity
        mat[-1, -2:] = np.array([-1, 1])  # backward difference
    elif order == 2:  # acceleration
        mat[-1, -3:] = np.array([1, -2, 1])  # backward difference
        mat[-2, -3:] = np.array([1, -2, 1])  # backward difference
    elif order == 3:  # jerk
        mat[-1, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-2, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-3, -4:] = np.array([-1, 3, -3, 1])  # backward difference

    # Return 
    return mat / (dt ** order)

def get_A_vel_acc_jerk(n=100,dt=1e-2):
    """
        Get matrices to compute velocities, accelerations, and jerks
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def smooth_optm_1d(
        traj,
        dt          = 0.1,
        x_init      = None,
        x_final     = None,
        vel_init    = None,
        vel_final   = None,
        x_lower     = None,
        x_upper     = None,
        vel_limit   = None,
        acc_limit   = None,
        jerk_limit  = None,
        idxs_remain = None,
        vals_remain = None,
        p_norm      = 2,
        verbose     = True,
    ):
    """
        1-D smoothing based on optimization
    """
    n = len(traj)
    A_pos = np.eye(n,n)
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    
    # Objective 
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x-traj,p_norm))
    
    # Equality constraints
    A_list,b_list = [],[]
    if x_init is not None:
        A_list.append(A_pos[0,:])
        b_list.append(x_init)
    if x_final is not None:
        A_list.append(A_pos[-1,:])
        b_list.append(x_final)
    if vel_init is not None:
        A_list.append(A_vel[0,:])
        b_list.append(vel_init)
    if vel_final is not None:
        A_list.append(A_vel[-1,:])
        b_list.append(vel_final)
    if idxs_remain is not None:
        A_list.append(A_pos[idxs_remain,:])
        if vals_remain is not None:
            b_list.append(vals_remain)
        else:
            b_list.append(traj[idxs_remain])

    # Inequality constraints
    C_list,d_list = [],[]
    if x_lower is not None:
        C_list.append(-A_pos)
        d_list.append(-x_lower*np.ones(n))
    if x_upper is not None:
        C_list.append(A_pos)
        d_list.append(x_upper*np.ones(n))
    if vel_limit is not None:
        C_list.append(A_vel)
        C_list.append(-A_vel)
        d_list.append(vel_limit*np.ones(n))
        d_list.append(vel_limit*np.ones(n))
    if acc_limit is not None:
        C_list.append(A_acc)
        C_list.append(-A_acc)
        d_list.append(acc_limit*np.ones(n))
        d_list.append(acc_limit*np.ones(n))
    if jerk_limit is not None:
        C_list.append(A_jerk)
        C_list.append(-A_jerk)
        d_list.append(jerk_limit*np.ones(n))
        d_list.append(jerk_limit*np.ones(n))
    constraints = []
    if A_list:
        A = np.vstack(A_list)
        b = np.hstack(b_list).squeeze()
        constraints.append(A @ x == b) 
    if C_list:
        C = np.vstack(C_list)
        d = np.hstack(d_list).squeeze()
        constraints.append(C @ x <= d)
    
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    # Return
    traj_smt = x.value

    # Null check
    if traj_smt is None and verbose:
        print ("[smooth_optm_1d] Optimization failed.")
    return traj_smt

def smooth_gaussian_1d(traj,sigma=5.0,mode='nearest',radius=5):
    """ 
        Smooting using Gaussian filter
    """
    traj_smt = gaussian_filter1d(input=traj,sigma=5.0,mode='nearest',radius=5)
    return traj_smt
    

def plot_traj_vel_acc_jerk(
        t,
        traj,
        traj_smt = None,
        figsize  = (6,6),
        title    = 'Trajectory',
        ):
    """ 
        Plot trajectory, velocity, acceleration, and jerk
    """
    n  = len(t)
    dt = t[1]-t[0]
    # Compute velocity, acceleration, and jerk
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    vel  = A_vel @ traj
    acc  = A_acc @ traj
    jerk = A_jerk @ traj
    if traj_smt is not None:
        vel_smt  = A_vel @ traj_smt
        acc_smt  = A_acc @ traj_smt
        jerk_smt = A_jerk @ traj_smt
    # Plot
    plt.figure(figsize=figsize)
    plt.subplot(4, 1, 1)
    plt.plot(t,traj,'.-',ms=1,color='k',lw=1/5,label='Trajectory')
    if traj_smt is not None:
        plt.plot(t,traj_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Trajectory')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 2)
    plt.plot(t,vel,'.-',ms=1,color='k',lw=1/5,label='Velocity')
    if traj_smt is not None:
        plt.plot(t,vel_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Velocity')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 3)
    plt.plot(t,acc,'.-',ms=1,color='k',lw=1/5,label='Acceleration')
    if traj_smt is not None:
        plt.plot(t,acc_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Acceleration')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 4)
    plt.plot(t,jerk,'.-',ms=1,color='k',lw=1/5,label='Jerk')
    if traj_smt is not None:
        plt.plot(t,jerk_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Jerk')
    plt.legend(fontsize=8,loc='upper right')
    plt.suptitle(title,fontsize=10)
    plt.subplots_adjust(hspace=0.2,top=0.95)
    plt.show()

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    if len(X1.shape) == 1: X1 = X1.reshape(-1,1)
    if len(X2.shape) == 1: X2 = X2.reshape(-1,1)
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def safe_chol(A,max_iter=100,eps=1e-20,verbose=False):
    """ 
        Safe Cholesky decomposition
    """
    A_use = A.copy()
    for iter in range(max_iter):
        try:
            L = np.linalg.cholesky(A_use)
            if verbose:
                print ("[safe_chol] Cholesky succeeded. iter:[%d] eps:[%.2e]"%(iter,eps))
            return L 
        except np.linalg.LinAlgError:
            A_use = A_use + eps*np.eye(A.shape[0])
            eps *= 10
    print ("[safe_chol] Cholesky failed. iter:[%d] eps:[%.2e]"%(iter,eps))
    return None

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
        x      = np.random.randn(100,5),
        x_min  = -np.ones(5),
        x_max  = np.ones(5),
        margin = 0.1,
    ):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def get_interp_const_vel_traj_nd(
        anchors, # [L x D]
        vel = 1.0,
        HZ  = 100,
        ord = np.inf,
    ):
    """
        Get linearly interpolated constant velocity trajectory
        Output is (times_interp,anchors_interp,times_anchor,idxs_anchor)
    """
    L = anchors.shape[0]
    D = anchors.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = anchors[tick-1,:],anchors[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp     = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    anchors_interp  = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D): # for each dim
        anchors_interp[:,d_idx] = np.interp(times_interp,times_anchor,anchors[:,d_idx])
    idxs_anchor = get_idxs_closest_ndarray(times_interp,times_anchor)
    return times_interp,anchors_interp,times_anchor,idxs_anchor

def interpolate_and_smooth_nd(
        anchors, # List or [N x D]
        HZ             = 50,
        vel_init       = 0.0,
        vel_final      = 0.0,
        x_lowers       = None, # [D]
        x_uppers       = None, # [D]
        vel_limit      = None, # [1]
        acc_limit      = None, # [1]
        jerk_limit     = None, # [1]
        vel_interp_max = d2r(180),
        vel_interp_min = d2r(10),
        n_interp       = 10,
        verbose        = False,
    ):
    """ 
        Interpolate anchors and smooth [N x D] anchors
    """
    if isinstance(anchors, list):
        # If 'anchors' is given as a list, make it an ndarray
        anchors = np.vstack(anchors)
    
    D = anchors.shape[1]
    vels = np.linspace(start=vel_interp_max,stop=vel_interp_min,num=n_interp)
    for v_idx,vel_interp in enumerate(vels):
        # First, interploate
        times,traj_interp,times_anchor,idxs_anchor = get_interp_const_vel_traj_nd(
            anchors = anchors,
            vel     = vel_interp,
            HZ      = HZ,
        )
        dt = times[1] - times[0]
        # Second, smooth
        traj_smt = np.zeros_like(traj_interp)
        is_success = True
        for d_idx in range(D):
            traj_d = traj_interp[:,d_idx]
            if x_lowers is not None: x_lower_d = x_lowers[d_idx]
            else: x_lower_d = None
            if x_uppers is not None: x_upper_d = x_uppers[d_idx]
            else: x_upper_d = None
            traj_smt_d = smooth_optm_1d(
                traj        = traj_d,
                dt          = dt,
                idxs_remain = idxs_anchor,
                vals_remain = anchors[:,d_idx],
                vel_init    = vel_init,
                vel_final   = vel_final,
                x_lower     = x_lower_d,
                x_upper     = x_upper_d,
                vel_limit   = vel_limit,
                acc_limit   = acc_limit,
                jerk_limit  = jerk_limit,
                p_norm      = 2,
                verbose     = False,
            )
            if traj_smt_d is None:
                is_success = False
                break
            # Append
            traj_smt[:,d_idx] = traj_smt_d

        # Check success
        if is_success:
            if verbose:
                print ("Optimization succeeded. vel_interp:[%.3f]"%(vel_interp))
            return times,traj_interp,traj_smt,times_anchor
        else:
            if verbose:
                print (" v_idx:[%d/%d] vel_interp:[%.2f] failed."%(v_idx,n_interp,vel_interp))
    
    # Optimization failed
    if verbose:
        print ("Optimization failed.")
    return times,traj_interp,traj_smt,times_anchor

def check_vel_acc_jerk_nd(
        times, # [L]
        traj, # [L x D]
        verbose = True,
        factor  = 1.0,
    ):
    """ 
        Check velocity, acceleration, jerk of n-dimensional trajectory
    """
    L,D = traj.shape[0],traj.shape[1]
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=len(times),dt=times[1]-times[0])
    vel_inits,vel_finals,max_vels,max_accs,max_jerks = [],[],[],[],[]
    for d_idx in range(D):
        traj_d = traj[:,d_idx]
        vel = A_vel @ traj_d
        acc = A_acc @ traj_d
        jerk = A_jerk @ traj_d
        vel_inits.append(vel[0])
        vel_finals.append(vel[-1])
        max_vels.append(np.abs(vel).max())
        max_accs.append(np.abs(acc).max())
        max_jerks.append(np.abs(jerk).max())

    # Print
    if verbose:
        print ("Checking velocity, acceleration, and jerk of a L:[%d]xD:[%d] trajectory (factor:[%.2f])."%
               (L,D,factor))
        for d_idx in range(D):
            print (" dim:[%d/%d]: v_init:[%.2e] v_final:[%.2e] v_max:[%.2f] a_max:[%.2f] j_max:[%.2f]"%
                   (d_idx,D,
                    factor*vel_inits[d_idx],factor*vel_finals[d_idx],
                    factor*max_vels[d_idx],factor*max_accs[d_idx],factor*max_jerks[d_idx])
                )
            
    # Return
    return vel_inits,vel_finals,max_vels,max_accs,max_jerks

class PID_ControllerClass(object):
    def __init__(
            self,
            name      = 'PID',
            k_p       = 0.01,
            k_i       = 0.0,
            k_d       = 0.001,
            dt        = 0.01,
            dim       = 1,
            dt_min    = 1e-6,
            out_min   = -np.inf,
            out_max   = np.inf,
            ANTIWU    = True,   # anti-windup
            out_alpha = 0.0,    # output EMA (0: no EMA)
        ):
        """
            Initialize PID Controller
        """
        self.name      = name
        self.k_p       = k_p
        self.k_i       = k_i
        self.k_d       = k_d
        self.dt        = dt
        self.dim       = dim
        self.dt_min    = dt_min
        self.out_min   = out_min
        self.out_max   = out_max
        self.ANTIWU    = ANTIWU
        self.out_alpha = out_alpha
        # Buffers
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = 0.0
        self.t_prev   = 0.0
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def reset(self,t_curr=0.0):
        """
            Reset PID Controller
        """
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = t_curr
        self.t_prev   = t_curr
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def update(
        self,
        t_curr  = None,
        x_trgt  = None,
        x_curr  = None,
        VERBOSE = False
        ):
        """
            Update PID controller
            u(t) = K_p e(t) + K_i int e(t) {dt} + K_d {de}/{dt}
        """
        if x_trgt is not None:
            self.x_trgt  = x_trgt
        if t_curr is not None:
            self.t_curr  = t_curr
        if x_curr is not None:
            self.x_curr  = x_curr
            # PID controller updates here
            self.dt       = max(self.t_curr - self.t_prev,self.dt_min)
            self.err_curr = self.x_trgt - self.x_curr     
            self.err_intg = self.err_intg + (self.err_curr*self.dt)
            self.err_diff = self.err_curr - self.err_prev
            
            if self.ANTIWU: # anti-windup
                self.err_out = self.err_curr * self.out_val
                self.err_intg[self.err_out<0.0] = 0.0
            
            if self.dt > self.dt_min:
                self.p_term   = self.k_p * self.err_curr
                self.i_term   = self.k_i * self.err_intg
                self.d_term   = self.k_d * self.err_diff / self.dt
                self.out_val  = np.clip(
                    a     = self.p_term + self.i_term + self.d_term,
                    a_min = self.out_min,
                    a_max = self.out_max)
                # Smooth the output control value using EMA
                self.out_val = self.out_alpha*self.out_val_prev + \
                    (1.0-self.out_alpha)*self.out_val
                self.out_val_prev = self.out_val

                if VERBOSE:
                    print ("cnt:[%d] t_curr:[%.5f] dt:[%.5f]"%
                           (self.cnt,self.t_curr,self.dt))
                    print (" x_trgt:   %s"%(self.x_trgt))
                    print (" x_curr:   %s"%(self.x_curr))
                    print (" err_curr: %s"%(self.err_curr))
                    print (" err_intg: %s"%(self.err_intg))
                    print (" p_term:   %s"%(self.p_term))
                    print (" i_term:   %s"%(self.i_term))
                    print (" d_term:   %s"%(self.d_term))
                    print (" out_val:  %s"%(self.out_val))
                    print (" err_out:  %s"%(self.err_out))
            # Backup
            self.t_prev   = self.t_curr
            self.err_prev = self.err_curr
        # Counter
        if (t_curr is not None) and (x_curr is not None):
            self.cnt = self.cnt + 1
            
    def out(self):
        """
            Get control output
        """
        return self.out_val.copy()

def sample_range(range):
    """ 
        Random sample from 'range':[L x 2] array where range[:,0] and range[:,1] contain min and max, respectively.
    """
    val_min = range[:,0]
    val_max = range[:,1]
    L = range.shape[0]
    val_sample = val_min + (val_max-val_min)*np.random.random_sample(L)
    return val_sample
    
def increase_tick(tick,L=None):
    """ 
        Increase tick until (L-1)
    """
    if L is not None:
        if tick < (L-1): 
            tick = tick + 1
    else:
        tick = tick + 1
    return tick 

def show_video_jnb(
        frames,
        HZ         = 50,
        width      = 500,
        downsample = False,
    ):
    """ 
        Save video in Jupyter Notebook
    """
    import mediapy as media
    media.show_video(images=frames,fps=HZ,width=width,downsample=downsample)
    
def smart_repr(var):
    if isinstance(var, float):
        return str(var)
    elif isinstance(var, np.ndarray):
        return f"ndarray with shape {var.shape}"
    elif isinstance(var, list) and len(var) == 0:
        return "[]"
    else:
        return str(var)
    
def get_consecutive_subarrays(array,min_element=1):
    """ 
        Get consecutive sub arrays from an array
    """
    split_points = np.where(np.diff(array) != 1)[0] + 1
    subarrays = np.split(array,split_points)    
    return [subarray for subarray in subarrays if len(subarray) >= min_element]
    
def get_contact_segments(
        secs,
        rtoe_traj,
        ltoe_traj,
        zvel_th     = 0.2, # toe z velocity threshold to detect contact
        min_seg_sec = 0.1, # minimum segment time (to filter out false-positve contact segments)
        smt_sigma   = 5.0, # Gaussian smoothing sigma
        smt_radius  = 5.0, # Gaussian smoothing radius (filter size)
        verbose     = True,
        plot        = True,
    ):
    """ 
        Get contact segments from right and left feet (or toe) trajectories
    """
    # Smooth and get velocity
    L,dt = len(secs),secs[1]-secs[0]
    HZ   = int(1/dt)
    A_vel,_,_ = get_A_vel_acc_jerk(n=L,dt=1/HZ)
    rtoe_traj_z,ltoe_traj_z = rtoe_traj[:,2],ltoe_traj[:,2]
    rtoe_traj_z_smt = smooth_gaussian_1d(traj=rtoe_traj_z,sigma=smt_sigma,mode='nearest',radius=smt_radius)
    ltoe_traj_z_smt = smooth_gaussian_1d(traj=ltoe_traj_z,sigma=smt_sigma,mode='nearest',radius=smt_radius)
    rtoe_veltraj_z,ltoe_veltraj_z = A_vel@rtoe_traj_z,A_vel@ltoe_traj_z
    rtoe_veltraj_z_smt,ltoe_veltraj_z_smt = A_vel@rtoe_traj_z_smt,A_vel@ltoe_traj_z_smt
    
    # Contact detection using smoothed trajectories
    ticks_rcontact = np.where(np.abs(rtoe_veltraj_z_smt) <= zvel_th)[0]
    ticks_lcontact = np.where(np.abs(ltoe_veltraj_z_smt) <= zvel_th)[0]
    
    # Get contact segments
    min_seg_len = int(min_seg_sec*HZ)
    rcontact_segs = get_consecutive_subarrays(ticks_rcontact,min_element=min_seg_len)
    lcontact_segs = get_consecutive_subarrays(ticks_lcontact,min_element=min_seg_len)
    
    # Print
    if verbose:
        print ("min_seg_sec:[%.2f]sec min_seg_len:[%d]"%(min_seg_sec,min_seg_len))
        print ("We have [%d] right contact segment(s)"%(len(rcontact_segs)))
        for seg_idx,seg in enumerate(rcontact_segs):
            print (" [%d] len:[%d]"%(seg_idx,len(seg)))
        print ("We have [%d] left contact segment(s)"%(len(lcontact_segs)))
        for seg_idx,seg in enumerate(lcontact_segs):
            print (" [%d] len:[%d]"%(seg_idx,len(seg)))
    
    # Plot results
    if plot:
        # Plot feet height
        plt.figure(figsize=(8,3))
        plt.plot(secs,rtoe_traj_z,'-',color='r',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Right Toe')
        plt.plot(secs,ltoe_traj_z,'-',color='b',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Left Toe')
        plt.plot(secs,rtoe_traj_z_smt,'-',color='r',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Right Toe')
        plt.plot(secs,ltoe_traj_z_smt,'-',color='b',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Left Toe')
        plt.plot(secs[ticks_rcontact],rtoe_traj_z[ticks_rcontact],
                linestyle='none',color='r',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
                label='Raw Right Contact')
        plt.plot(secs[ticks_lcontact],ltoe_traj_z[ticks_lcontact],
                linestyle='none',color='b',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
                label='Raw Left Contact')
        markers = ['o','v','^','<','>','s','p']
        for seg_idx,rseg in enumerate(rcontact_segs):
            plt.plot(secs[rseg],rtoe_traj_z[rseg],
                    linestyle='none',color='r',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Right Contact [%d]'%(seg_idx))
        for seg_idx,lseg in enumerate(lcontact_segs):
            plt.plot(secs[lseg],ltoe_traj_z[lseg],
                    linestyle='none',color='b',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Left Contact [%d]'%(seg_idx))
        plt.title('Toe Height',fontsize=8)
        plt.xlabel('Time (sec)',fontsize=8)
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0,fontsize=6)
        plt.tight_layout(); plt.show()

        # Plot feet velocity
        plt.figure(figsize=(8,3))
        plt.plot(secs,zvel_th*np.ones(L),'--',color='k',lw=1,marker='none',mfc='none',ms=3,mew=0.5)
        plt.plot(secs,-zvel_th*np.ones(L),'--',color='k',lw=1,marker='none',mfc='none',ms=3,mew=0.5)
        plt.plot(secs,rtoe_veltraj_z,'-',color='r',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Right Toe')
        plt.plot(secs,ltoe_veltraj_z,'-',color='b',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Left Toe')
        plt.plot(secs,rtoe_veltraj_z_smt,'-',color='r',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Right Toe')
        plt.plot(secs,ltoe_veltraj_z_smt,'-',color='b',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Left Toe')
        plt.plot(secs[ticks_rcontact],rtoe_veltraj_z[ticks_rcontact],
                linestyle='none',color='r',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
                label='Raw Right Contact')
        plt.plot(secs[ticks_lcontact],ltoe_veltraj_z[ticks_lcontact],
                linestyle='none',color='b',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
                label='Raw Left Contact')
        for seg_idx,rseg in enumerate(rcontact_segs):
            plt.plot(secs[rseg],rtoe_veltraj_z_smt[rseg],
                    linestyle='none',color='r',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Right Contact [%d]'%(seg_idx))
        for seg_idx,lseg in enumerate(lcontact_segs):
            plt.plot(secs[lseg],ltoe_veltraj_z_smt[lseg],
                    linestyle='none',color='b',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Left Contact [%d]'%(seg_idx))
        plt.title('Toe Velocity',fontsize=8)
        plt.xlabel('Time (sec)',fontsize=8)
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0,fontsize=6)
        plt.tight_layout(); plt.show()
        
    # Return contact segments
    return rcontact_segs,lcontact_segs

def save_png(img,png_path,verbose=False):
    """ 
        Save image
    """
    directory = os.path.dirname(png_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose:
            print ("[%s] generated."%(directory))
    # Save to png
    plt.imsave(png_path,img)
    if verbose:
        print ("[%s] saved."%(png_path))
        
def fit_xy_circle(xy):
    x,y = xy[:,0],xy[:,1]
    A = np.column_stack((2*x, 2*y, np.ones(x.size)))
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc,yc = c[0],c[1]
    r  = np.sqrt(c[2] + xc**2 + yc**2)
    center = c[:2] # [x]
    radius = r # [1]
    return center,radius        

def exclude_overlapping_pcd_within_list(pcd_list):
    """ 
        Assuming that 'pcd_list' is a list of point clouds (i.e., xyzs),
        this function will excluding overlapping point clouds within the list.
    """
    # Save the original points
    all_points = []
    original_indices = []
    for i, pcd in enumerate(pcd_list):
        all_points.append(pcd)
        original_indices.append(np.full(len(pcd), i))
    all_points = np.vstack(all_points)
    original_indices = np.concatenate(original_indices)
    # Get unique points
    unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    
    # Resort
    sorted_indices = np.argsort(unique_indices)
    unique_points = unique_points[sorted_indices]
    original_indices = original_indices[unique_indices][sorted_indices]
    
    # Get the list
    unique_pcd_list = []
    for i in range(len(pcd_list)):
        unique_pcd_list.append(unique_points[original_indices == i])
        
    # Return
    return unique_pcd_list

class TicTocClass(object):
    """
        Tic toc
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            Initialize
        """
        self.name        = name
        self.time_start  = time.time()
        self.time_end    = time.time()
        self.print_every = print_every

    def tic(self):
        """
            Tic
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=0,VERBOSE=True):
        """
            Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if VERBOSE:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if (cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))
                    
def is_point_in_polygon(point,polygon):
    """
        Is the point inside the polygon
    """
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
        Is the point feasible w.r.t. obstacle list
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False                    
    
def is_point_to_point_connectable(point1,point2,obs_list):
    """
        Is the line connecting two points connectable
    """
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False

def np_uv(vec):
    """
        Get unit vector
    """
    x = np.array(vec)
    len = np.linalg.norm(x+1e-8)
    if len <= 1e-6:
        return np.array([0,0,1])
    else:
        return x/len
        
def uv_T_joi(T_joi,joi_fr,joi_to):
    """ 
        Get unit vector between to JOI poses
    """
    return np_uv(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def len_T_joi(T_joi,joi_fr,joi_to):
    """ 
        Get length between two JOI poses
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))
    
def mean_rotation_matrix(R_list):
    # Convert the list of rotation matrices into a single 3x3xN numpy array
    R = np.stack(R_list, axis=2)
    # Compute the mean of the rotation matrices
    M = np.mean(R, axis=2)
    # Perform SVD on the mean matrix
    U, _, Vt = np.linalg.svd(M)
    # Compute the mean rotation matrix
    R_mean = np.dot(U, Vt)
    return R_mean

def np2torch(x_np,device): 
    return torch.tensor(x_np,dtype=torch.float32,device=device)

def torch2np(x_torch): 
    return x_torch.detach().cpu().numpy()