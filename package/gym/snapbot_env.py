import numpy as np
""" 
    Assume that the main notebook called 'sys.path.append('../../package/helper/')'
"""
from transformation import r2rpy

class SnapbotGymClass():
    """ 
        Snapbot gym 
    """
    def __init__(self,env,HZ=50,history_total_sec=2.0,history_intv_sec=0.1,VERBOSE=True):
        """
            Initialize
        """
        self.env               = env # MuJoCo environment
        self.HZ                = HZ
        self.dt                = 1/self.HZ
        self.history_total_sec = history_total_sec # history in seconds
        self.n_history         = int(self.HZ*self.history_total_sec) # number of history
        self.history_intv_sec  = history_intv_sec
        self.history_intv_tick = int(self.HZ*self.history_intv_sec) # interval between state in history
        self.history_ticks     = np.arange(0,self.n_history,self.history_intv_tick)
        
        self.mujoco_nstep      = self.env.HZ // self.HZ # nstep for MuJoCo step
        self.VERBOSE           = VERBOSE
        self.tick              = 0
        
        self.name              = env.name
        self.state_prev        = self.get_state()
        self.action_prev       = self.sample_action()
        
        # Dimensions
        self.state_dim         = len(self.get_state())
        self.state_history     = np.zeros((self.n_history,self.state_dim))
        self.tick_history      = np.zeros((self.n_history,1))
        self.o_dim             = len(self.get_observation())
        self.a_dim             = env.n_ctrl
        
        if VERBOSE:
            print ("[%s] Instantiated"%
                   (self.name))
            print ("   [info] dt:[%.4f] HZ:[%d], env-HZ:[%d], mujoco_nstep:[%d], state_dim:[%d], o_dim:[%d], a_dim:[%d]"%
                   (self.dt,self.HZ,self.env.HZ,self.mujoco_nstep,self.state_dim,self.o_dim,self.a_dim))
            print ("   [history] total_sec:[%.2f]sec, n:[%d], intv_sec:[%.2f]sec, intv_tick:[%d]"%
                   (self.history_total_sec,self.n_history,self.history_intv_sec,self.history_intv_tick))
            print ("   [history] ticks:%s"%(self.history_ticks))
        
    def get_state(self):
        """
            Get state (33)
            : Current state consists of 
                1) current joint position (8)
                2) current joint velocity (8)
                3) torso rotation (9)
                4) torso height (1)
                5) torso y value (1)
                6) contact info (8)
        """
        # Joint position
        qpos = self.env.data.qpos[self.env.ctrl_qpos_idxs] # joint position
        # Joint velocity
        qvel = self.env.data.qvel[self.env.ctrl_qvel_idxs] # joint velocity
        # Torso rotation matrix flattened
        R_torso_flat = self.env.get_R_body(body_name='torso').reshape((-1)) # torso rotation
        # Torso height
        p_torso = self.env.get_p_body(body_name='torso') # torso position
        torso_height = np.array(p_torso[2]).reshape((-1))
        # Torso y value
        torso_y_value = np.array(p_torso[1]).reshape((-1))
        # Contact information
        contact_info = np.zeros(self.env.n_sensor)
        contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
        contact_info[contact_idxs] = 1.0 # 1 means contact occurred
        # Concatenate information
        state = np.concatenate([
            qpos,
            qvel/10.0, # scale
            R_torso_flat,
            torso_height,
            torso_y_value,
            contact_info
        ])
        return state
    
    def get_observation(self):
        """
            Get observation 
        """
        
        # Sparsely accumulated history vector 
        state_history_sparse = self.state_history[self.history_ticks,:]
        
        # Concatenate information
        obs = np.concatenate([
            state_history_sparse
        ])
        return obs.flatten()

    def sample_action(self):
        """
            Sample action (8)
        """
        a_min  = self.env.ctrl_ranges[:,0]
        a_max  = self.env.ctrl_ranges[:,1]
        action = a_min + (a_max-a_min)*np.random.rand(len(a_min))
        return action
        
    def step(self,a,max_time=np.inf):
        """
            Step forward
        """
        # Increse tick
        self.tick = self.tick + 1
        
        # Previous torso position and yaw angle in degree
        p_torso_prev       = self.env.get_p_body('torso')
        R_torso_prev       = self.env.get_R_body('torso')
        yaw_torso_deg_prev = np.degrees(r2rpy(R_torso_prev)[2])
        
        # Run simulation for 'mujoco_nstep' steps
        self.env.step(ctrl=a,nstep=self.mujoco_nstep)
        
        # Current torso position and yaw angle in degree
        p_torso_curr       = self.env.get_p_body('torso')
        R_torso_curr       = self.env.get_R_body('torso')
        yaw_torso_deg_curr = np.degrees(r2rpy(R_torso_curr)[2])
        
        # Compute the done signal
        ROLLOVER = (np.dot(R_torso_curr[:,2],np.array([0,0,1]))<0.0)
        if (self.get_sim_time() >= max_time) or ROLLOVER:
            d = True
        else:
            d = False
        
        # Compute forward reward
        x_diff = p_torso_curr[0] - p_torso_prev[0] # x-directional displacement
        r_forward = x_diff/self.dt
        
        # Check self-collision (excluding 'floor')
        p_contacts,f_contacts,geom1s,geom2s,_,_ = self.env.get_contact_info(must_exclude_prefix='floor')
        if len(geom1s) > 0: # self-collision occurred
            SELF_COLLISION = 1
            r_collision    = -10.0
        else:
            SELF_COLLISION = 0
            r_collision    = 0.0
            
        # Survival reward
        if ROLLOVER:
            r_survive = -10.0
        else:
            r_survive = 0.01
        
        # Heading reward
        heading_vec = R_torso_curr[:,0] # x direction
        r_heading = 0.01*np.dot(heading_vec,np.array([1,0,0]))
        if r_heading < 0.0:
            r_heading = r_heading*100.0 # focus more on penalizing going wrong direction
            
        # Lane keeping
        lane_deviation = p_torso_curr[1] # y-directional displacement
        r_lane = -np.abs(lane_deviation)*0.5
        
        # Compute reward
        r = np.array(r_forward+r_collision+r_survive+r_heading+r_lane)
        
        # Accumulate state history (update 'state_history')
        self.accumulate_state_history()
        
        # Next observation 'accumulate_state_history' should be called before calling 'get_observation'
        o_prime = self.get_observation()
        
        # Other information
        info = {'yaw_torso_deg_prev':yaw_torso_deg_prev,'yaw_torso_deg_curr':yaw_torso_deg_curr,
                'x_diff':x_diff,'SELF_COLLISION':SELF_COLLISION,
                'r_forward':r_forward,'r_collision':r_collision,'r_survive':r_survive,
                'r_heading':r_heading,'r_lane':r_lane}
        
        # Return
        return o_prime,r,d,info
    
    def render(
            self,
            TRACK_TORSO      = True,
            PLOT_WORLD_COORD = True,
            PLOT_TORSO_COORD = True,
            PLOT_SENSOR      = True,
            PLOT_CONTACT     = True,
            PLOT_TIME        = True,
        ):
        """
            Render
        """
        # Change lookat
        if TRACK_TORSO:
            p_lookat = self.env.get_p_body('torso')
            self.env.set_viewer(lookat=p_lookat)
        # World coordinate
        if PLOT_WORLD_COORD:
            self.env.plot_T(p=np.zeros(3),R=np.eye(3,3),plot_axis=True,axis_len=1.0,axis_width=0.0025)
        # Plot snapbot base
        if PLOT_TORSO_COORD:
            p_torso,R_torso = self.env.get_pR_body(body_name='torso') # update viewer
            self.env.plot_T(p=p_torso,R=R_torso,plot_axis=True,axis_len=0.25,axis_width=0.0025)
        # Plot contact sensors
        if PLOT_SENSOR:
            contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
            for idx in contact_idxs:
                sensor_name = self.env.sensor_names[idx]
                p_sensor = self.env.get_p_sensor(sensor_name)
                self.env.plot_sphere(p=p_sensor,r=0.02,rgba=[1,0,0,0.2])
        # Plot contact info
        if PLOT_CONTACT:
            self.env.plot_contact_info()
        # Plot time and tick on top of torso
        if PLOT_TIME:
            self.env.plot_T(p=p_torso+0.25*R_torso[:,2],R=np.eye(3,3),
                       plot_axis=False,label='[%.2f]sec'%(self.env.get_sim_time()))
        # Do render
        self.env.render()

    def reset(self):
        """
            Reset
        """
        # Reset parameters
        self.tick = 0
        # Reset env
        self.env.reset(step=True)
        # Reset history
        self.state_history = np.zeros((self.n_history,self.state_dim))
        self.tick_history  = np.zeros((self.n_history,1))
        # Get observation
        o = self.get_observation()
        return o
        
    def init_viewer(self):
        """
            Initialize viewer
        """
        self.env.init_viewer(distance=3.0)
        
    def close_viewer(self):
        """
            Close viewer
        """
        self.env.close_viewer()
        
    def get_sim_time(self):
        """
            Get time (sec)
        """
        return self.env.get_sim_time()
    
    def is_viewer_alive(self):
        """
            Check whether the viewer is alive
        """
        return self.env.is_viewer_alive()
    
    def accumulate_state_history(self):
        """
            Get state history
        """
        state = self.get_state()
        # Accumulate 'state' and 'tick'
        self.state_history[1:,:] = self.state_history[:-1,:]
        self.state_history[0,:]  = state
        self.tick_history[1:,:]  = self.tick_history[:-1,:]
        self.tick_history[0,:]   = self.tick
        
    def viewer_pause(self):
        """
            Viewer pause
        """
        self.env.viewer_pause()
        
    def grab_image(self,resize_rate=1.0,interpolation=0):
        """
            Grab image
        """
        return self.env.grab_image(rsz_rate=resize_rate,interpolation=interpolation)
