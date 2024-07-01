import os,mujoco,time,cv2,copy,glfw
import numpy as np
from mujoco_custom_viewer import MujocoMinimalViewer
"""
sys.path.append('../../package/helper/')
 => this should be called before calling 'from mujoco_parser import MuJoCoParserClass' 
"""
from transformation import (
    t2p,
    t2r,
    rpy2r,
    r2quat,
    pr2t,
    r2w,
    get_rotation_matrix_from_two_points,
)
from utility import (
    compute_view_params,
    meters2xyz,
    get_idxs,
    trim_scale,
    get_colors,
    d2r,
    interpolate_and_smooth_nd,
)
from slider import MultiSliderClass

class MuJoCoParserClass(object):
    """
        MuJoCo Parser
    """
    def __init__(
            self,
            name         = 'Robot',
            rel_xml_path = None,
            verbose      = True,
        ):
        """ 
            Initialize MuJoCo parser
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.verbose      = verbose

        # Constants
        self.tick              = 0
        self.render_tick       = 0
        self.use_mujoco_viewer = False
        
        # Parse xml file
        if self.rel_xml_path is not None:
            self._parse_xml()

        # Print
        if self.verbose:
            self.print_info()

        # Reset
        self.reset(step=True)

    def _parse_xml(self):
        """ 
            Parse xml file
        """
        self.full_xml_path    = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
        self.model            = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data             = mujoco.MjData(self.model)
        self.dt               = self.model.opt.timestep
        self.HZ               = int(1/self.dt)

        # State and action space
        self.n_qpos           = self.model.nq # number of states
        self.n_qvel           = self.model.nv # number of velocities (dimension of tangent space)
        self.n_qacc           = self.model.nv # number of accelerations (dimension of tangent space)

        # Geometry
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_idx)
                                 for geom_idx in range(self.model.ngeom)]
        
        # Body
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body_idx)
                                 for body_idx in range(self.n_body)]
        self.body_masses      = self.model.body_mass # (kg)
        self.body_total_mass  = self.body_masses.sum()
        
        self.parent_body_names = []
        for b_idx in range(self.n_body):
            parent_id = self.model.body_parentid[b_idx]
            parent_body_name = self.body_names[parent_id]
            self.parent_body_names.append(parent_body_name)
        
        # Degree of Freedom
        self.n_dof            = self.model.nv # degree of freedom (=number of columns of Jacobian)
        self.dof_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_DOF,dof_idx)
                                 for dof_idx in range(self.n_dof)]

        # Joint
        self.n_joint          = self.model.njnt # number of joints 
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
                                 for joint_idx in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges

        # Free joint
        self.free_joint_idxs  = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.free_joint_names = [self.joint_names[joint_idx] for joint_idx in self.free_joint_idxs]
        self.n_free_joint     = len(self.free_joint_idxs)

        # Revolute Joint
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins

        # Prismatic Joint
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.pri_joint_idxs]
        self.n_pri_joint      = len(self.pri_joint_idxs)
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins

        # Controls
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) 
                                 for ctrl_idx in range(self.n_ctrl)]
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        self.ctrl_mins        = self.ctrl_ranges[:,0]
        self.ctrl_maxs        = self.ctrl_ranges[:,1]
        self.ctrl_gears       = self.model.actuator_gear[:,0] # gears

        # qpos and qvel indices attached to the controls
        """ 
        # Usage
        self.env.data.qpos[self.env.ctrl_qpos_idxs] # joint position
        self.env.data.qvel[self.env.ctrl_qvel_idxs] # joint velocity
        """
        self.ctrl_qpos_idxs = []
        self.ctrl_qpos_names = []
        self.ctrl_qpos_mins = []
        self.ctrl_qpos_maxs = []
        self.ctrl_qvel_idxs = []
        self.ctrl_types = []
        for ctrl_idx in range(self.n_ctrl):
            # transmission (joint) index attached to an actuator, we assume that there is just one joint attached
            joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0] 
            # joint position attached to control
            self.ctrl_qpos_idxs.append(self.model.jnt_qposadr[joint_idx])
            self.ctrl_qpos_names.append(self.joint_names[joint_idx])
            self.ctrl_qpos_mins.append(self.joint_ranges[joint_idx,0])
            self.ctrl_qpos_maxs.append(self.joint_ranges[joint_idx,1])
            # joint velocity attached to control
            self.ctrl_qvel_idxs.append(self.model.jnt_dofadr[joint_idx])
            # Check types
            trntype = self.model.actuator_trntype[ctrl_idx]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                self.ctrl_types.append('JOINT')
            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                self.ctrl_types.append('TENDON')
            else:
                self.ctrl_types.append('UNKNOWN(trntype=%d)'%(trntype))

        # Sensor
        self.n_sensor         = self.model.nsensor
        self.sensor_names     = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_idx)
                                 for sensor_idx in range(self.n_sensor)]
        
        # Site
        self.n_site           = self.model.nsite
        self.site_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,site_idx)
                                 for site_idx in range(self.n_site)]

    def print_info(self):
        """ 
            Print model information
        """
        print ("name:[%s] dt:[%.3f] HZ:[%d]"%(self.name,self.dt,self.HZ))
        print ("n_qpos:[%d] n_qvel:[%d] n_qacc:[%d] n_ctrl:[%d]"%(self.n_qpos,self.n_qvel,self.n_qacc,self.n_ctrl))

        print ("")
        print ("n_body:[%d]"%(self.n_body))
        for body_idx,body_name in enumerate(self.body_names):
            body_mass = self.body_masses[body_idx]
            print (" [%d/%d] [%s] mass:[%.2f]kg"%(body_idx,self.n_body,body_name,body_mass))
        print ("body_total_mass:[%.2f]kg"%(self.body_total_mass))
        
        print ("")
        print ("n_geom:[%d]"%(self.n_geom))
        print ("geom_names:%s"%(self.geom_names))

        print ("")
        print ("n_joint:[%d]"%(self.n_joint))
        for joint_idx,joint_name in enumerate(self.joint_names):
            print (" [%d/%d] [%s] axis:%s"%
                   (joint_idx,self.n_joint,joint_name,self.model.joint(joint_idx).axis))
        # print ("joint_types:[%s]"%(self.joint_types))
        # print ("joint_ranges:[%s]"%(self.joint_ranges))

        print ("")
        print ("n_dof:[%d] (=number of rows of Jacobian)"%(self.n_dof))
        for dof_idx,dof_name in enumerate(self.dof_names):
            joint_name= self.joint_names[self.model.dof_jntid[dof_idx]]
            body_name= self.body_names[self.model.dof_bodyid[dof_idx]]
            print (" [%d/%d] [%s] attached joint:[%s] body:[%s]"%
                   (dof_idx,self.n_dof,dof_name,joint_name,body_name))
        
        print ("\nFree joint information. n_free_joint:[%d]"%(self.n_free_joint))
        for idx,free_joint_name in enumerate(self.free_joint_names):
            body_name_attached = self.body_names[self.model.joint(self.free_joint_idxs[idx]).bodyid[0]]
            print (" [%d/%d] [%s] body_name_attached:[%s]"%
                   (idx,self.n_free_joint,free_joint_name,body_name_attached))
            
        print ("\nRevolute joint information. n_rev_joint:[%d]"%(self.n_rev_joint))
        for idx,rev_joint_name in enumerate(self.rev_joint_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f]"%
                   (idx,self.n_rev_joint,rev_joint_name,self.rev_joint_mins[idx],self.rev_joint_maxs[idx]))

        print ("\nPrismatic joint information. n_pri_joint:[%d]"%(self.n_pri_joint))
        for idx,pri_joint_name in enumerate(self.pri_joint_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f]"%
                   (idx,self.n_pri_joint,pri_joint_name,self.pri_joint_mins[idx],self.pri_joint_maxs[idx]))
            
        print ("\nControl information. n_ctrl:[%d]"%(self.n_ctrl))
        for idx,ctrl_name in enumerate(self.ctrl_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f] gear:[%.2f] type:[%s]"%
                   (idx,self.n_ctrl,ctrl_name,self.ctrl_mins[idx],self.ctrl_maxs[idx],
                    self.ctrl_gears[idx],self.ctrl_types[idx]))
        print ("")
        print ("n_sensor:[%d]"%(self.n_sensor))
        print ("sensor_names:%s"%(self.sensor_names))
        print ("n_site:[%d]"%(self.n_site))
        print ("site_names:%s"%(self.site_names))

    def reset(self,step=False):
        """
            Reset
        """
        mujoco.mj_resetData(self.model,self.data) # reset data
        
        if step:
            mujoco.mj_step(self.model,self.data)
            # mujoco.mj_forward(self.model,self.data) # forward <= is this necessary?
        
        # Reset ticks
        self.tick        = 0
        self.render_tick = 0
        # Reset wall time
        self.init_sim_time  = self.data.time
        self.init_wall_time = time.time()
        # Others
        self.xyz_double_click = None 

    def init_viewer(
            self,
            title         = None,
            width         = 1400,
            height        = 1000,
            hide_menu     = True,
            fontscale     = mujoco.mjtFontScale.mjFONTSCALE_200.value,
            azimuth       = 170, # None,
            distance      = 5.0, # None,
            elevation     = -20, # None,
            lookat        = [0.01,0.11,0.5], # None,
            transparent   = None,
            contactpoint  = None,
            contactwidth  = None,
            contactheight = None,
            contactrgba   = None,
            joint         = None,
            jointlength   = None,
            jointwidth    = None,
            jointrgba     = None,
            geomgroup_0   = None,
            geomgroup_1   = None,
            geomgroup_2   = None,
            update        = False,
            maxgeom       = 10000,
        ):
        """ 
            Initialize viewer
        """
        self.use_mujoco_viewer = True
        if title is None: title = self.name

        self.viewer = MujocoMinimalViewer(
            self.model,
            self.data,
            mode       = 'window',
            title      = title,
            width      = width,
            height     = height,
            hide_menus = hide_menu,
            maxgeom    = maxgeom,
        )
        self.viewer.ctx = mujoco.MjrContext(self.model,fontscale)
        
        # Set viewer
        self.set_viewer(
            azimuth       = azimuth,
            distance      = distance,
            elevation     = elevation,
            lookat        = lookat,
            transparent   = transparent,
            contactpoint  = contactpoint,
            contactwidth  = contactwidth,
            contactheight = contactheight,
            contactrgba   = contactrgba,
            joint         = joint,
            jointlength   = jointlength,
            jointwidth    = jointwidth,
            jointrgba     = jointrgba,
            geomgroup_0   = geomgroup_0,
            geomgroup_1   = geomgroup_1,
            geomgroup_2   = geomgroup_2,
            update        = update,
        )

    def set_viewer(
            self,
            azimuth       = None,
            distance      = None,
            elevation     = None,
            lookat        = None,
            transparent   = None,
            contactpoint  = None,
            contactwidth  = None,
            contactheight = None,
            contactrgba   = None,
            joint         = None,
            jointlength   = None,
            jointwidth    = None,
            jointrgba     = None,
            geomgroup_0   = None,
            geomgroup_1   = None,
            geomgroup_2   = None,
            update        = False,
        ):
        """ 
            Set MuJoCo Viewer
        """
        # Basic viewer setting (azimuth, distance, elevation, and lookat)
        if azimuth is not None: self.viewer.cam.azimuth = azimuth
        if distance is not None: self.viewer.cam.distance = distance
        if elevation is not None: self.viewer.cam.elevation = elevation
        if lookat is not None: self.viewer.cam.lookat = lookat
        # Make dynamic geoms more transparent
        if transparent is not None: 
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
        # Contact point
        if contactpoint is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contactpoint
        if contactwidth is not None: self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None: self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None: self.model.vis.rgba.contactpoint = contactrgba
        # Joint
        if joint is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = joint
        if jointlength is not None: self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None: self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None: self.model.vis.rgba.joint = jointrgba
        # Geom group
        if geomgroup_0 is not None: self.viewer.vopt.geomgroup[0] = geomgroup_0
        if geomgroup_1 is not None: self.viewer.vopt.geomgroup[1] = geomgroup_1
        if geomgroup_2 is not None: self.viewer.vopt.geomgroup[2] = geomgroup_2
        # Render to update settings
        if update:
            mujoco.mj_forward(self.model,self.data) 
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)

    def get_viewer_cam_info(self,verbose=False):
        """
            Get viewer cam information
        """
        azimuth   = self.viewer.cam.azimuth
        distance  = self.viewer.cam.distance
        elevation = self.viewer.cam.elevation
        lookat    = self.viewer.cam.lookat.copy()
        if verbose:
            print ("azimuth:[%.2f] distance:[%.2f] elevation:[%.2f] lookat:%s]"%
                   (azimuth,distance,elevation,lookat))
        return azimuth,distance,elevation,lookat
    
    def is_viewer_alive(self):
        """
            Check whether a viewer is alive
        """
        return self.viewer.is_alive
    
    def close_viewer(self):
        """
            Close viewer
        """
        self.use_mujoco_viewer = False
        self.viewer.close()

    def render(self):
        """
            Render
        """
        if self.use_mujoco_viewer:
            self.viewer.render()
        else:
            print ("[%s] Viewer NOT initialized."%(self.name))

    def loop_every(self,HZ=None,tick_every=None):
        """
            Loop every
        """
        # tick = int(self.get_sim_time()/self.dt)
        FLAG = False
        if HZ is not None:
            FLAG = (self.tick-1)%(int(1/self.dt/HZ))==0
        if tick_every is not None:
            FLAG = (self.tick-1)%(tick_every)==0
        return FLAG
    
    def step(self,ctrl=None,ctrl_idxs=None,joint_names=None,nstep=1,increase_tick=True):
        """
            Forward dynamics
        """
        if ctrl is not None:
            if joint_names is not None: # if 'joint_names' is not None, it overrides 'ctrl_idxs'
                ctrl_idxs = self.get_idxs_step(joint_names=joint_names)
            if ctrl_idxs is None: self.data.ctrl[:] = ctrl
            else: self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)
        if increase_tick: self.tick = self.tick + 1

    def forward(self,q=None,joint_idxs=None,joint_names=None,increase_tick=True):
        """
            Forward kinematics
        """
        if q is not None:
            if joint_names is not None: # if 'joint_names' is not None, it override 'joint_idxs'
                joint_idxs = self.get_idxs_fwd(joint_names=joint_names)
            if joint_idxs is not None: 
                self.data.qpos[joint_idxs] = q
            else: self.data.qpos = q
        mujoco.mj_forward(self.model,self.data)
        if increase_tick: self.tick = self.tick + 1

    def increase_tick(self):
        """ 
            Increase tick
        """
        self.tick = self.tick + 1

    def get_state(self):
        """ 
            Get MuJoCo state (tick, time, qpos, qvel, act)
            ...
            The state vector in MuJoCo is:
                x = (mjData.time, mjData.qpos, mjData.qvel, mjData.act)
            Next we turn to the controls and applied forces. The control vector in MuJoCo is
                u = (mjData.ctrl, mjData.qfrc_applied, mjData.xfrc_applied)
            These quantities specify control signals (mjData.ctrl) for the actuators defined in the model, 
            or directly apply forces and torques specified in joint space (mjData.qfrc_applied) 
            or in Cartesian space (mjData.xfrc_applied).
        """
        state = {
            'tick':self.tick,
            'time':self.data.time,
            'qpos':self.data.qpos.copy(), # [self.model.nq]
            'qvel':self.data.qvel.copy(), # [self.model.nv]
            'act':self.data.act.copy(),
        }
        return state
    
    def store_state(self):
        """ 
            Store MuJoCo state
        """
        state = self.get_state()
        self.state_store = copy.deepcopy(state) # deep copy

    def restore_state(self):
        """ 
            Restore MuJoCo state
        """
        state = self.state_store
        self.set_state(
            qpos = state['qpos'],
            qvel = state['qvel'],
            act  = state['act'],
        )
        mujoco.mj_forward(self.model,self.data)
    
    def set_state(
            self,
            tick = None,
            time = None,
            qpos = None,
            qvel = None,
            act  = None, # used for simulating tendons and muscles
            ctrl = None,
            step = False
        ):
        """ 
            Set MuJoCo state
        """
        if tick is not None: self.tick = tick
        if time is not None: self.data.time = time
        if qpos is not None: self.data.qpos = qpos.copy()
        if qvel is not None: self.data.qvel = qvel.copy()
        if act is not None: self.data.act = act.copy()
        if ctrl is not None: self.data.ctrl = ctrl.copy()
        # Forward dynamics
        if step: 
            mujoco.mj_step(self.model,self.data)

    def solve_inverse_dynamics(self,qacc=None):
        """ 
            Solve Inverse Dynamics
        """
        if qacc is None:
            qacc = np.zeros(self.n_qacc)
        # Set desired qacc
        self.data.qacc = qacc.copy()
        # Store state
        self.store_state()
        # Solve inverse dynamics
        mujoco.mj_inverse(self.model,self.data)
        # Restore state
        self.restore_state()
        # Return  
        """
            Output is 'qfrc_inverse'
            This is the force that must have acted on the system in order to achieve the observed acceleration 'mjData.qacc'.
        """
        qfrc_inverse = self.data.qfrc_inverse # [n_qacc]
        return qfrc_inverse.copy()

    def set_p_base_body(self,body_name='base',p=np.array([0,0,0])):
        """ 
            Set position of base body
        """
        jntadr  = self.model.body(body_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr:qposadr+3] = p

    def set_R_base_body(self,body_name='base',R=rpy2r(np.radians([0,0,0]))):
        """ 
            Set Rotation of base body
        """
        jntadr  = self.model.body(body_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr+3:qposadr+7] = r2quat(R)
        
    def set_T_base_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None):
        """ 
            Set Pose of base body
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        self.set_p_base_body(body_name=body_name,p=p)
        self.set_R_base_body(body_name=body_name,R=R)
        

    def set_p_body(self,body_name='base',p=np.array([0,0,0])):
        """ 
            Set position of body (not base body)
        """
        self.model.body(body_name).pos = p

    def set_geom_color(
            self,
            body_names_to_color   = None,
            body_names_to_exclude = ['world'],
            body_names_to_exclude_including = [],
            rgba                  = [0.75,0.95,0.15,1.0],
            rgba_list             = None,
        ):
        """
            Set body color
        """
        def should_exclude(x, exclude_list):
            for exclude in exclude_list:
                if exclude in x:
                    return True
            return False
        
        if body_names_to_color is None: # default is to color all geometries
            body_names_to_color = self.body_names
        for idx,body_name in enumerate(body_names_to_color): # for all bodies
            if body_name in body_names_to_exclude: # exclude specific bodies
                continue 
            if should_exclude(body_name,body_names_to_exclude_including): 
                # exclude body_name including ones in 'body_names_to_exclude_including'
                continue
            body_idx = self.body_names.index(body_name)
            geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
            for geom_idx in geom_idxs: # for geoms attached to the body
                if rgba_list is None:
                    self.model.geom(geom_idx).rgba = rgba
                else:
                    self.model.geom(geom_idx).rgba = rgba_list[idx]
            
    def get_sim_time(self,init_flag=False):
        """
            Get simulation time (sec)
        """
        if init_flag:
            self.init_sim_time = self.data.time
        elapsed_time = self.data.time - self.init_sim_time
        return elapsed_time
    
    def reset_sim_time(self):
        """
            Reset simulation time (sec)
        """
        self.init_sim_time = self.data.time
    
    def get_wall_time(self,init_flag=False):
        """ 
            Get wall clock time
        """
        if init_flag:
            self.init_wall_time = time.time()
        elapsed_time = time.time() - self.init_wall_time # second
        return elapsed_time
    
    def grab_rgbd_img(self):
        """
            Grab RGB and Depth images
        """
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # flip up-down

        # Rescale depth image
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img
    
    def get_T_viewer(self):
        """
            Get viewer pose
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth]))
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3))
        return T_viewer
    
    def get_pcd_from_depth_img(self,depth_img,fovy=45):
        """
            Get point cloud data from depth image
        """
        # Get camera pose
        T_viewer = self.get_T_viewer()

        # Camera intrinsic
        img_height = depth_img.shape[0]
        img_width = depth_img.shape[1]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                            (0,focal_scaling,img_height/2),
                            (0,0,1)))

        # Estimate 3D point from depth image
        xyz_img = meters2xyz(depth_img,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]

        # To world coordinate
        xyzone_world_transpose = T_viewer @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        xyz_world = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]

        xyz_img_world = xyz_world.reshape(depth_img.shape[0],depth_img.shape[1],3)

        return xyz_world,xyz_img,xyz_img_world
    
    def get_egocentric_rgbd_pcd(
            self,
            p_ego            = None,
            p_trgt           = None,
            rsz_rate_for_pcd = None,
            rsz_rate_for_img = None,
            fovy             = None,
            restore_view     = True,
        ):
        """
            Get egocentric 1) RGB image, 2) Depth image, 3) Point Cloud Data
        """
        if restore_view:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos=p_ego,target_pos=p_trgt,up_vector=np.array([0,0,1]))
            self.set_viewer(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
                update    = True,
            )
        
        # Grab RGB and depth image
        rgb_img,depth_img = self.grab_rgbd_img() # get rgb and depth images

        # Resize depth image for reducing point clouds
        if rsz_rate_for_pcd is not None:
            h_rsz         = int(depth_img.shape[0]*rsz_rate_for_pcd)
            w_rsz         = int(depth_img.shape[1]*rsz_rate_for_pcd)
            depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_rsz = depth_img

        # Get PCD
        if fovy is None:
            if len(self.model.cam_fovy): fovy = 45.0 # if cam is not defined, use 45deg (default value)
            else: fovy = self.model.cam_fovy[0] # otherwise use the fovy of the first camera
        pcd,xyz_img,xyz_img_world = self.get_pcd_from_depth_img(depth_img_rsz,fovy=fovy) # [N x 3]

        # Resize rgb_image and depth_img (optional)
        if rsz_rate_for_img is not None:
            h = int(rgb_img.shape[0]*rsz_rate_for_img)
            w = int(rgb_img.shape[1]*rsz_rate_for_img)
            rgb_img   = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img,(w,h),interpolation=cv2.INTER_NEAREST)

        # Restore view
        if restore_view:
            # Restore camera information
            self.set_viewer(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
                update    = True,
            )
        return rgb_img,depth_img,pcd,xyz_img,xyz_img_world
    
    def grab_image(self,rsz_rate=None,interpolation=cv2.INTER_NEAREST):
        """
            Grab the rendered iamge
        """
        img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
        mujoco.mjr_readPixels(img, None,self.viewer.viewport,self.viewer.ctx)
        img = np.flipud(img) # flip image
        # Resize
        if rsz_rate is not None:
            h = int(img.shape[0]*rsz_rate)
            w = int(img.shape[1]*rsz_rate)
            img = cv2.resize(img,(w,h),interpolation=interpolation)
        # Backup
        if img.sum() > 0:
            self.grab_image_backup = img
        if img.sum() == 0: # use backup instead
            img = self.grab_image_backup
        return img.copy()
    
    def get_body_names(self,prefix='',excluding='world'):
        """
            Get body names with prefix
        """
        body_names = [x for x in self.body_names if x is not None and x.startswith(prefix) and excluding not in x]
        return body_names
    
    def get_p_body(self,body_name):
        """
            Get body position
        """
        return self.data.body(body_name).xpos.copy()

    def get_R_body(self,body_name):
        """
            Get body rotation matrix
        """
        return self.data.body(body_name).xmat.reshape([3,3]).copy()
    
    def get_T_body(self,body_name):
        """
            Get body pose
        """
        p_body = self.get_p_body(body_name=body_name)
        R_body = self.get_R_body(body_name=body_name)
        return pr2t(p_body,R_body)

    def get_pR_body(self,body_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R
    
    def get_p_joint(self,joint_name):
        """
            Get joint position
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_p_body(self.body_names[body_id])

    def get_R_joint(self,joint_name):
        """
            Get joint rotation matrix
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_R_body(self.body_names[body_id])
    
    def get_pR_joint(self,joint_name):
        """
            Get joint position and rotation matrix
        """
        p = self.get_p_joint(joint_name)
        R = self.get_R_joint(joint_name)
        return p,R

    def get_p_geom(self,geom_name):
        """ 
            Get geom position
        """
        return self.data.geom(geom_name).xpos
    
    def get_R_geom(self,geom_name):
        """ 
            Get geom rotation
        """
        return self.data.geom(geom_name).xmat.reshape((3,3))
    
    def get_pR_geom(self,geom_name):
        """
            Get geom position and rotation matrix
        """
        p = self.get_p_geom(geom_name)
        R = self.get_R_geom(geom_name)
        return p,R
    
    def get_p_sensor(self,sensor_name):
        """
             Get sensor position
        """
        sensor_id = self.model.sensor(sensor_name).id # get sensor ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
        sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
        p = self.data.site(site_name).xpos.copy() # get the position of the site
        return p
    
    def get_R_sensor(self,sensor_name):
        """
             Get sensor position
        """
        sensor_id = self.model.sensor(sensor_name).id
        sensor_objtype = self.model.sensor_objtype[sensor_id]
        sensor_objid = self.model.sensor_objid[sensor_id]
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid)
        R = self.data.site(site_name).xmat.reshape([3,3]).copy()
        return R
    
    def get_pR_sensor(self,sensor_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return p,R

    def get_sensor_value(self,sensor_name):
        """
            Read sensor value
        """
        data = self.data.sensor(sensor_name).data
        return data.copy()

    def get_sensor_values(self,sensor_names=None):
        """
            Read multiple sensor values
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        data = np.array([self.get_sensor_value(sensor_name) for sensor_name in self.sensor_names]).squeeze()
        return data.copy()

    def plot_T(
            self,
            p           = np.array([0,0,0]),
            R           = np.eye(3),
            T           = None,
            plot_axis   = True,
            axis_len    = 1.0,
            axis_width  = 0.005,
            axis_rgba   = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
            print_xyz   = False,
        ):
        """ 
            Plot coordinate axes
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
            
        if plot_axis:
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,0.9]
                rgba_y = [0.0,1.0,0.0,0.9]
                rgba_z = [0.0,0.0,1.0,0.9]
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p + R_x[:,2]*axis_len/2
            if print_xyz: axis_label = 'X-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = axis_label,
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            if print_xyz: axis_label = 'Y-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = axis_label,
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            if print_xyz: axis_label = 'Z-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = axis_label,
            )

        if plot_sphere:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')

        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_sphere(self,p,r,rgba=[1,1,1,1],label=''):
        """
            Plot sphere
        """
        self.viewer.add_marker(
            pos   = p,
            size  = [r,r,r],
            rgba  = rgba,
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label)
        
    def plot_spheres(self,ps,r,rgba=[1,1,1,1],label=''):
        """ 
            Plot spheres
        """
        for p in ps:
            self.plot_sphere(p=p,r=r,rgba=rgba,label=label)
        
    def plot_box(
            self,
            p    = np.array([0,0,0]),
            R    = np.eye(3),
            xlen = 1.0,
            ylen = 1.0,
            zlen = 1.0,
            rgba = [0.5,0.5,0.5,0.5]
        ):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen,ylen,zlen],
            rgba  = rgba,
            label = ''
        )
    
    def plot_capsule(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size  = [r,r,h],
            rgba  = rgba,
            label = ''
        )
        
    def plot_cylinder(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,h],
            rgba  = rgba,
            label = ''
        )
    
    def plot_ellipsoid(self,p=np.array([0,0,0]),R=np.eye(3),rx=1.0,ry=1.0,rz=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size  = [rx,ry,rz],
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,h*2],
            rgba  = rgba,
            label = ''
        )
        
    def plot_line(self,p=np.array([0,0,0]),R=np.eye(3),h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = h,
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow_fr2to(self,p_fr,p_to,r=1.0,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = ''
        )

    def plot_line_fr2to(self,p_fr,p_to,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = ''
        )
    
    def plot_cylinder_fr2to(self,p_fr,p_to,r=0.01,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = ''
        )
        
    def plot_traj(
            self,
            traj,
            rgba          = [1,0,0,1],
            plot_line     = True,
            plot_cylinder = False,
            plot_sphere   = False,
            cylinder_r    = 0.01,
            sphere_r      = 0.01,
        ):
        """ 
            Plot trajectory
        """
        L = traj.shape[0]
        for idx in range(L-1):
            p_fr = traj[idx,:]
            p_to = traj[idx+1,:]
            if plot_line:
                self.plot_line_fr2to(p_fr=p_fr,p_to=p_to,rgba=rgba)
            if plot_cylinder:
                self.plot_cylinder_fr2to(p_fr=p_fr,p_to=p_to,r=cylinder_r,rgba=rgba)
        if plot_sphere:
            for idx in range(L):
                p = traj[idx,:]
                self.plot_sphere(p=p,r=sphere_r,rgba=rgba)
        
    def plot_text(self,p,label=''):
        """ 
            Plot text
        """
        self.viewer.add_marker(
            pos   = p,
            size  = [0.0001,0.0001,0.0001],
            rgba  = [1,1,1,0.01],
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label,
        )

    def plot_time(self,p=np.array([0,0,1]),post_str=''):
        """ 
            Plot text
        """
        self.plot_text(
            p     = p,
            label = "[%d] sim_time:[%.2f]sec wall_time:[%.2f]sec %s"%
                    (self.tick,self.get_sim_time(),self.get_wall_time(),post_str),
        )

    def plot_body_T(
            self,
            body_name,
            plot_axis   = True,
            axis_len    = 0.2,
            axis_width  = 0.01,
            axis_rgba   = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
        ):
        """
            Plot coordinate axes on a body
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(
            p,
            R,
            plot_axis   = plot_axis,
            axis_len    = axis_len,
            axis_width  = axis_width,
            axis_rgba   = axis_rgba,
            plot_sphere = plot_sphere,
            sphere_r    = sphere_r,
            sphere_rgba = sphere_rgba,
            label       = label,
        )
        
    def plot_joint_T(
            self,
            joint_name,
            plot_axis  = True,
            axis_len   = 1.0,
            axis_width = 0.01,
            axis_rgba  = None,
            label      = None,
        ):
        """
            Plot coordinate axes on a joint
        """
        p,R = self.get_pR_joint(joint_name=joint_name)
        self.plot_T(
            p,
            R,
            plot_axis  = plot_axis,
            axis_len   = axis_len,
            axis_width = axis_width,
            axis_rgba  = axis_rgba,
            label      = label,
        )
        
    def plot_bodies_T(
            self,
            body_names            = None,
            body_names_to_exclude = [],
            body_names_to_exclude_including = [],
            print_name            = False,
            axis_len              = 0.05,
            axis_width            = 0.005,
        ):
        """ 
            Plot bodies T
        """
        def should_exclude(x, exclude_list):
            for exclude in exclude_list:
                if exclude in x:
                    return True
            return False
        
        if body_names is None:
            body_names = self.body_names
            
        for body_idx,body_name in enumerate(body_names):
            if body_name in body_names_to_exclude: continue
            
            if should_exclude(body_name,body_names_to_exclude_including): 
                # exclude body_name including ones in 'body_names_to_exclude_including'
                continue
            
            if print_name:
                label = '[%d] %s'%(body_idx,body_name)
            else:
                label = ''
            self.plot_body_T(
                body_name  = body_name,
                plot_axis  = True,
                axis_len   = axis_len,
                axis_width = axis_width,
                label      = label,
            )
            
    def plot_links_between_bodies(
            self,
            parent_body_names_to_exclude = ['world'],
            r                            = 0.005,
            rgba                         = (0.0,0.0,0.0,0.5),
        ):
        """ 
            Plot links between bodies
        """
        for body_idx,body_name in enumerate(self.body_names):
            parent_body_name = self.parent_body_names[body_idx]
            if parent_body_name in parent_body_names_to_exclude: continue
            if body_name is None: continue
            
            self.plot_cylinder_fr2to(
                p_fr = self.get_p_body(body_name=parent_body_name),
                p_to = self.get_p_body(body_name=body_name),
                r    = r,
                rgba = rgba,
            )

    def plot_joint_axis(
            self,
            axis_len    = 0.1,
            axis_r      = 0.01,
            joint_names = None,
            alpha       = 0.2,
        ):
        """ 
            Plot revolute joint axis 
        """
        rev_joint_idxs  = self.rev_joint_idxs
        rev_joint_names = self.rev_joint_names

        if joint_names is not None:
            idxs = get_idxs(self.rev_joint_names,joint_names)
            rev_joint_idxs_to_use  = rev_joint_idxs[idxs]
            rev_joint_names_to_use = [rev_joint_names[i] for i in idxs]
        else:
            rev_joint_idxs_to_use  = rev_joint_idxs
            rev_joint_names_to_use = rev_joint_names

        for rev_joint_idx,rev_joint_name in zip(rev_joint_idxs_to_use,rev_joint_names_to_use):
            axis_joint      = self.model.jnt_axis[rev_joint_idx]
            p_joint,R_joint = self.get_pR_joint(joint_name=rev_joint_name)
            axis_world      = R_joint@axis_joint
            axis_rgba       = np.append(np.eye(3)[:,np.argmax(axis_joint)],alpha)
            self.plot_arrow_fr2to(
                p_fr = p_joint,
                p_to = p_joint+axis_len*axis_world,
                r    = axis_r,
                rgba = axis_rgba
            )

    def get_contact_body_names(self):
        """ 
            Get contacting body names
        """
        contact_body_names = []
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
        return contact_body_names

    def get_contact_info(self,must_include_prefix=None,must_exclude_prefix=None):
        """
            Get contact information
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        body1s = []
        body2s = []
        for c_idx in range(self.data.ncon):
            contact   = self.data.contact[c_idx]
            # Contact position and frame orientation
            p_contact = contact.pos # contact position
            R_frame   = contact.frame.reshape(( 3,3))
            # Contact force
            f_contact_local = np.zeros(6,dtype=np.float64)
            mujoco.mj_contactForce(self.model,self.data,0,f_contact_local)
            f_contact = R_frame @ f_contact_local[:3] # in the global coordinate
            # Contacting geoms
            contact_geom1 = self.geom_names[contact.geom1]
            contact_geom2 = self.geom_names[contact.geom2]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            # Append
            if must_include_prefix is not None:
                if (contact_geom1[:len(must_include_prefix)] == must_include_prefix) or (contact_geom2[:len(must_include_prefix)] == must_include_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            elif must_exclude_prefix is not None:
                if (contact_geom1[:len(must_exclude_prefix)] != must_exclude_prefix) and (contact_geom2[:len(must_exclude_prefix)] != must_exclude_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            else:
                p_contacts.append(p_contact)
                f_contacts.append(f_contact)
                geom1s.append(contact_geom1)
                geom2s.append(contact_geom2)
                body1s.append(contact_body1)
                body2s.append(contact_body2)
        return p_contacts,f_contacts,geom1s,geom2s,body1s,body2s

    def print_contact_info(self,must_include_prefix=None):
        """ 
            Print contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            print ("Tick:[%d] Body contact:[%s]-[%s]"%(self.tick,body1,body2))

    def plot_arrow_contact(self,p,uv,r_arrow=0.03,h_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
            Plot arrow
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_arrow,r_arrow,h_arrow],
            rgba  = rgba,
            label = label
        )

    def plot_joints(
            self,
            joint_names      = None,
            plot_axis        = True,
            axis_len         = 0.1,
            axis_width       = 0.01,
            axis_rgba        = None,
            plot_joint_names = False,
        ):
        """ 
            Plot joint names
        """
        if joint_names is None:
            joint_names = self.joint_names
        for joint_name in joint_names:
            if joint_name is not None:
                if plot_joint_names:
                    label = joint_name
                else:
                    label = None
                self.plot_joint_T(
                    joint_name,
                    plot_axis  = plot_axis,
                    axis_len   = axis_len,
                    axis_width = axis_width,
                    axis_rgba  = axis_rgba,
                    label      = label,
                )

    def plot_contact_info(
            self,
            must_include_prefix = None,
            r_arrow             = 0.005,
            h_arrow             = 0.1,
            rgba_contact        = [1,0,0,1],
            r_sphere            = 0.02,
            plot_arrow          = True,
            plot_sphere         = False,
            print_contact_body  = False,
            print_contact_geom  = False,
            verbose             = False
        ):
        """
            Plot contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        # Render contact informations
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            f_norm = np.linalg.norm(f_contact)
            f_uv   = f_contact / (f_norm+1e-8)
            # h_arrow = 0.3 # f_norm*0.05
            if plot_arrow:
                self.plot_arrow_contact(
                    p       = p_contact,
                    uv      = f_uv,
                    r_arrow = r_arrow,
                    h_arrow = h_arrow,
                    rgba    = rgba_contact,
                    label   = '',
                )
                self.plot_arrow_contact(
                    p       = p_contact,
                    uv      = -f_uv,
                    r_arrow = r_arrow,
                    h_arrow = h_arrow,
                    rgba    = rgba_contact,
                    label   = '',
                )
            if plot_sphere: 
                # contact_label = '[%s]-[%s]'%(body1,body2)
                contact_label = ''
                self.plot_sphere(p=p_contact,r=r_sphere,rgba=rgba_contact,label=contact_label)
            if print_contact_body:
                label = '[%s]-[%s]'%(body1,body2)
            elif print_contact_geom:
                label = '[%s]-[%s]'%(geom1,geom2)
            else:
                label = '' 
        # Print
        if verbose:
            self.print_contact_info(must_include_prefix=must_include_prefix)

    def get_idxs_fwd(self,joint_names):
        """ 
            Get indices for using env.forward()
            Example)
            env.forward(q=q,joint_idxs=idxs_fwd) # <= HERE
        """
        return [self.model.joint(jname).qposadr[0] for jname in joint_names]
    
    def get_idxs_jac(self,joint_names):
        """ 
            Get indices for solving inverse kinematics
            Example)
            J,ik_err = env.get_ik_ingredients(...)
            dq = env.damped_ls(J,ik_err,stepsize=1,eps=1e-2,th=np.radians(1.0))
            q = q + dq[idxs_jac] # <= HERE
        """
        return [self.model.joint(jname).dofadr[0] for jname in joint_names]
    
    def get_idxs_step(self,joint_names):
        """ 
            Get indices for using env.step()
            Example)
            env.step(ctrl=q,ctrl_idxs=idxs_step) # <= HERE
        """
        return [self.ctrl_qpos_names.index(jname) for jname in joint_names]
    
    def get_qpos(self):
        """ 
            Get joint positions
        """
        return self.data.qpos.copy() # [n_qpos]
    
    def get_qvel(self):
        """ 
            Get joint velocities
        """
        return self.data.qvel.copy() # [n_qvel]
    
    def get_qacc(self):
        """ 
            Get joint accelerations
        """
        return self.data.qacc.copy() # [n_qacc]

    def get_qpos_joint(self,joint_name):
        """
            Get joint position
        """
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qvel_joint(self,joint_name):
        """
            Get joint velocity
        """
        addr = self.model.joint(joint_name).dofadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        if L > 1: L = 6
        qvel = self.data.qvel[addr:addr+L]
        return qvel
    
    def get_qpos_joints(self,joint_names):
        """
            Get multiple joint positions from 'joint_names'
        """
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_qvel_joints(self,joint_names):
        """
            Get multiple joint velocities from 'joint_names'
        """
        return np.array([self.get_qvel_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_ctrl(self,ctrl_names):
        """ 
            Get control values
        """
        idxs = get_idxs(self.ctrl_names,ctrl_names)
        return np.array([self.data.ctrl[idx] for idx in idxs]).squeeze()
    
    def set_qpos_joints(self,joint_names,qpos):
        """ 
            Set joint positions
        """
        joint_idxs = self.get_idxs_fwd(joint_names)
        self.data.qpos[joint_idxs] = qpos
        mujoco.mj_forward(self.model,self.data)

    def set_ctrl(self,ctrl_names,ctrl,nstep=1):
        """ 
        """
        ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
        self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)
        
    def viewer_pause(self):
        """
            Viewer pause
        """
        self.viewer._paused = True
        
    def viewer_resume(self):
        """
            Viewer resume
        """
        self.viewer._paused = False
    
    def get_viewer_mouse_xy(self):
        """
            Get viewer mouse (x,y)
        """
        viewer_mouse_xy = np.array([self.viewer._last_mouse_x,self.viewer._last_mouse_y])
        return viewer_mouse_xy
    
    def get_xyz_left_double_click(self):
        """ 
            Get xyz location of double click
        """
        flag_click = False
        if self.viewer._left_double_click_pressed: # left double click
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd()
            self.xyz_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._left_double_click_pressed = False
            flag_click = True
        return self.xyz_double_click,flag_click
    
    def is_left_double_clicked(self):
        """ 
            Check left double click
        """
        if self.viewer._left_double_click_pressed: # left double click
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd()
            self.xyz_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._left_double_click_pressed = False # toggle flag
            return True 
        else:
            return False
    
    def get_body_name_closest(self,xyz,body_names=None,verbose=False):
        """
            Get the closest body name to xyz
        """
        if body_names is None:
            body_names = self.body_names
        dists = np.zeros(len(body_names))
        p_body_list = []
        for body_idx,body_name in enumerate(body_names):
            p_body = self.get_p_body(body_name=body_name)
            dist = np.linalg.norm(p_body-xyz)
            dists[body_idx] = dist # append
            p_body_list.append(p_body) # append
        idx_min = np.argmin(dists)
        body_name_closest = body_names[idx_min]
        p_body_closest = p_body_list[idx_min]
        if verbose:
            print ("[%s] selected"%(body_name_closest))
        return body_name_closest,p_body_closest
    
    # Inverse kinematics
    def get_J_body(self,body_name):
        """
            Get Jocobian matrices of a body
        """
        J_p = np.zeros((3,self.n_dof)) # nv: nDoF
        J_R = np.zeros((3,self.n_dof))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_J_geom(self,geom_name):
        """
            Get Jocobian matrices of a geom
        """
        J_p = np.zeros((3,self.n_dof)) # nv: nDoF
        J_R = np.zeros((3,self.n_dof))
        mujoco.mj_jacGeom(self.model,self.data,J_p,J_R,self.data.geom(geom_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_ik_ingredients(
            self,
            body_name = None,
            geom_name = None,
            p_trgt    = None,
            R_trgt    = None,
            IK_P      = True,
            IK_R      = True,
        ):
        """
            Get IK ingredients
        """
        if body_name is not None:
            J_p,J_R,J_full = self.get_J_body(body_name=body_name)
            p_curr,R_curr = self.get_pR_body(body_name=body_name)
        if geom_name is not None:
            J_p,J_R,J_full = self.get_J_geom(geom_name=geom_name)
            p_curr,R_curr = self.get_pR_geom(geom_name=geom_name)
        if (body_name is not None) and (geom_name is not None):
            print ("[get_ik_ingredients] body_name:[%s] geom_name:[%s] are both not None!"%(body_name,geom_name))
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err
    
    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
            Dampled least square for IK
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq

    def onestep_ik(
            self,
            body_name  = None,
            geom_name  = None,
            p_trgt     = None,
            R_trgt     = None,
            IK_P       = True,
            IK_R       = True,
            joint_idxs = None,
            stepsize   = 1,
            eps        = 1e-1,
            th         = 5*np.pi/180.0,
        ):
        """
            Solve IK for a single step
        """
        J,err = self.get_ik_ingredients(
            body_name = body_name,
            geom_name = geom_name,
            p_trgt    = p_trgt,
            R_trgt    = R_trgt,
            IK_P      = IK_P,
            IK_R      = IK_R,
            )
        dq = self.damped_ls(J,err,stepsize=stepsize,eps=eps,th=th)
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        q = self.get_q(joint_idxs=joint_idxs)
        q = q + dq[joint_idxs]
        # FK
        self.forward(q=q,joint_idxs=joint_idxs)
        return q, err
    
    def is_key_pressed(self,char=None,chars=None,upper=True):
        """ 
            Check keyboard pressed (high-level function calling 'check_key_pressed()')
        """
        if self.viewer._is_key_pressed:
            self.viewer._is_key_pressed = False
            return self.check_key_pressed(char=char,chars=chars,upper=upper)
        else:
            return False

    def check_key_pressed(self,char=None,chars=None,upper=True):
        """
            Check keyboard pressed from a character (e.g., 'a','b','1', or ['a','b','c'])
        """
        # Check a single character
        if char is not None:
            if upper: char = char.upper()
            if self.get_key_pressed() == char:
                return True
        
        # Check a list of characters
        if chars is not None:
            for _char in chars:
                if upper: _char = _char.upper()
                if self.get_key_pressed() == _char:
                    return True
        
        # (default) Return False
        return False
        
    def get_key_pressed(self,to_int=False):
        """ 
            Get keyboard pressed
        """
        if self.viewer._key_pressed is None: return None
        char = chr(self.viewer._key_pressed)
        if to_int: char = int(char) # to integer
        return char
    
    def animate_free_fall(
            self,
            plot_every  = 1,
            transparent = True,
            azimuth     = 170,
            distance    = 5,
            elevation   = -27,
            lookat      = [0.01, 0.11, 0.8],
        ):
        """ 
            Animate the simple free fall motion
        """
        self.reset(step=True)
        self.init_viewer(
            transparent = transparent,
            azimuth     = azimuth,
            distance    = distance,
            elevation   = elevation,
            lookat      = lookat,
        )
        while self.is_viewer_alive():    
            # Step
            self.step()
            # Render
            if self.loop_every(tick_every=plot_every):
                self.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
                self.plot_time()
                self.plot_contact_info(r_arrow=0.005,h_arrow=0.1,plot_sphere=False,verbose=False)
                self.render()
        self.close_viewer()

    def animate_kinematic_slider_control(
            self,
            joint_names = None,
            plot_every  = 10,
            transparent = True,
            azimuth     = 170,
            distance    = 5,
            elevation   = -27,
            lookat      = [0.01, 0.11, 0.8],
            axis_len    = 0.1,
            axis_r      = 0.01,
            plot_org_T  = True,
            plot_link   = True,
        ):
        """ 
            Animate basic kinematic slider control
        """
        
        # Reset env
        self.reset(step=True)

        # Configuration
        if joint_names is None:
            joint_names = self.rev_joint_names # default is to use revolute joints
        idxs = get_idxs(self.rev_joint_names,joint_names)
        init_qpos = self.get_qpos_joints(joint_names=joint_names)
        resolution = np.deg2rad(0.1)
        sliders = MultiSliderClass(
            n_slider      = len(joint_names),
            title         = 'Sliders for [%s] Control'%(self.name),
            window_width  = 600,
            window_height = 800,
            x_offset      = 100,
            y_offset      = 100,
            slider_width  = 450,
            label_texts   = [self.rev_joint_names[idx] for idx in idxs],
            slider_mins   = [self.rev_joint_mins[idx] for idx in idxs],
            slider_maxs   = [self.rev_joint_maxs[idx] for idx in idxs],
            slider_vals   = init_qpos,
            resolution    = resolution,
            verbose       = False,
        )
        idxs_fwd = self.get_idxs_fwd(joint_names=joint_names)

        # Initialize viewer
        self.init_viewer(
            transparent = transparent,
            azimuth     = azimuth,
            distance    = distance,
            elevation   = elevation,
            lookat      = lookat,
        )
        
        # Loop
        while self.is_viewer_alive():
            # Update
            sliders.update() # update slider
            self.forward(q=sliders.get_slider_values(),joint_idxs=idxs_fwd)
            # Render
            if self.loop_every(tick_every=plot_every):
                if plot_org_T:
                    self.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
                self.plot_time()
                self.plot_contact_info(r_arrow=0.005,h_arrow=0.1,plot_sphere=False,verbose=False)
                self.plot_joint_axis(
                    axis_len    = axis_len,
                    axis_r      = axis_r,
                    joint_names = joint_names,
                )
                # Plot links
                if plot_link:
                    self.plot_links_between_bodies(
                        parent_body_names_to_exclude=['world'],
                        r    = 0.0025,
                        rgba = (0,0,0,1)
                    )
                self.render()

        # Close
        self.close_viewer()
        sliders.close()

    def animate_dynamic_slider_control(
            self,
            ctrl_names  = None,
            init_ctrl   = None,
            plot_every  = 10,
            transparent = True,
            azimuth     = 170,
            distance    = 5,
            elevation   = -27,
            lookat      = [0.01, 0.11, 0.8],
            axis_len    = 0.1,
            axis_r      = 0.01,
        ):
        """ 
            Animate basic dynamic slider control
        """
        # Reset
        self.reset(step=True)

        # Configuration
        if ctrl_names is None:
            ctrl_names = self.ctrl_names # default is to use revolute joints
        idxs = get_idxs(self.ctrl_names,ctrl_names)
        if init_ctrl is None:
            init_ctrl = self.get_ctrl(ctrl_names=ctrl_names)
        resolution = 0.1
        sliders = MultiSliderClass(
            n_slider      = len(ctrl_names),
            title         = 'Sliders for [%s] Control'%(self.name),
            window_width  = 600,
            window_height = 800,
            x_offset      = 300,
            y_offset      = 200,
            slider_width  = 450,
            label_texts   = [self.ctrl_names[idx] for idx in idxs],
            slider_mins   = [self.ctrl_ranges[idx,0] for idx in idxs],
            slider_maxs   = [self.ctrl_ranges[idx,1] for idx in idxs],
            slider_vals   = init_ctrl,
            resolution    = resolution,
            verbose       = False,
        )

        # Initialize viewer
        self.init_viewer(
            transparent = transparent,
            azimuth     = azimuth,
            distance    = distance,
            elevation   = elevation,
            lookat      = lookat,
        )

        # Loop
        while self.is_viewer_alive():
            # Update
            sliders.update() # update slider
            self.step(ctrl=sliders.get_slider_values(),ctrl_idxs=idxs)
            # Render
            if self.loop_every(tick_every=plot_every):
                self.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
                self.plot_time()
                self.plot_contact_info(r_arrow=0.005,h_arrow=0.1,plot_sphere=False,verbose=False)
                self.plot_joint_axis(
                    axis_len    = axis_len,
                    axis_r      = axis_r,
                    joint_names = None,
                )
                self.render()

        # Close
        self.close_viewer()
        sliders.close()

# Inverse kinematics helper
def init_ik_info():
    """
        Initialize IK information
        Usage:
        ik_info = init_ik_info()
        ...
        add_ik_info(ik_info,body_name='BODY_NAME',p_trgt=P_TRGT,R_trgt=R_TRGT)
        ...
        for ik_tick in range(max_ik_tick):
            dq,ik_err_stack = get_dq_from_ik_info(
                env = env,
                ik_info = ik_info,
                stepsize = 1,
                eps = 1e-2,
                th = np.radians(10.0),
                joint_idxs_jac = joint_idxs_jac,
            )
            qpos = env.get_qpos()
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            if np.linalg.norm(ik_err_stack) < 0.05: break
    """
    ik_info = {
        'body_names':[],
        'geom_names':[],
        'p_trgts':[],
        'R_trgts':[],
        'n_trgt':0,
    }
    return ik_info

def add_ik_info(
        ik_info,
        body_name = None,
        geom_name = None,
        p_trgt    = None,
        R_trgt    = None,
    ):
    """ 
        Add IK information
    """
    ik_info['body_names'].append(body_name)
    ik_info['geom_names'].append(geom_name)
    ik_info['p_trgts'].append(p_trgt)
    ik_info['R_trgts'].append(R_trgt)
    ik_info['n_trgt'] = ik_info['n_trgt'] + 1

def get_dq_from_ik_info(
        env,
        ik_info,
        stepsize       = 1,
        eps            = 1e-2,
        th             = np.radians(1.0),
        joint_idxs_jac = None,
    ):
    """
        Get delta q from augmented Jacobian method
    """
    J_list,ik_err_list = [],[]
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None
        J,ik_err = env.get_ik_ingredients(
            body_name = ik_body_name,
            geom_name = ik_geom_name,
            p_trgt    = ik_p_trgt,
            R_trgt    = ik_R_trgt,
            IK_P      = IK_P,
            IK_R      = IK_R,
        )
        J_list.append(J)
        ik_err_list.append(ik_err)

    J_stack      = np.vstack(J_list)
    ik_err_stack = np.hstack(ik_err_list)

    # Select Jacobian columns that are within the joints to use
    if joint_idxs_jac is not None:
        J_stack_backup = J_stack.copy()
        J_stack = np.zeros_like(J_stack)
        J_stack[:,joint_idxs_jac] = J_stack_backup[:,joint_idxs_jac]

    # Compute dq from damped least square
    dq = env.damped_ls(J_stack,ik_err_stack,stepsize=stepsize,eps=eps,th=th)
    return dq,ik_err_stack

def plot_ik_info(
        env,ik_info,
        axis_len=0.05,axis_width=0.005,
        sphere_r=0.01
        ):
    """
        Plot IK information
    """
    colors = get_colors(cmap_name='gist_rainbow',n_color=ik_info['n_trgt'])
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        color = colors[ik_idx]
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None

        if ik_body_name is not None:
            # Plot current 
            env.plot_body_T(
                body_name   = ik_body_name,
                plot_axis   = IK_R,
                axis_len    = axis_len,
                axis_width  = axis_width,
                plot_sphere = IK_P,
                sphere_r    = sphere_r,
                sphere_rgba = color,
                label       = '' # ''/ik_body_name
            )
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='') 
                env.plot_line_fr2to(p_fr=env.get_p_body(body_name=ik_body_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R:
                p_curr = env.get_p_body(body_name=ik_body_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            
        if ik_geom_name is not None:
            # Plot current 
            env.plot_geom_T(
                geom_name   = ik_geom_name,
                plot_axis   = IK_R,
                axis_len    = axis_len,
                axis_width  = axis_width,
                plot_sphere = IK_P,
                sphere_r    = sphere_r,
                sphere_rgba = color,
                label       = '' # ''/ik_geom_name
            )
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='') 
                env.plot_line_fr2to(p_fr=env.get_p_geom(geom_name=ik_geom_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R:
                p_curr = env.get_p_geom(geom_name=ik_geom_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)

def solve_ik(
        env,
        joint_names_for_ik,
        body_name_trgt,
        q_init          = None, # IK start from the initial pose
        p_trgt          = None,
        R_trgt          = None,
        max_ik_tick     = 100,
        ik_err_th       = 1e-2,
        restore_state   = True,
        ik_stepsize     = 1.0,
        ik_eps          = 1e-2,
        ik_th           = np.radians(1.0),
        verbose         = False,
        verbose_warning = True,
        reset_env       = False,
        render          = False,
        render_every    = 1,
    ):
    """ 
        Solve Inverse Kinematics
    """
    # Reset
    if reset_env:
        env.reset()
    if render:
        env.init_viewer()
    # Joint indices
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_idxs_fwd = env.get_idxs_fwd(joint_names=joint_names_for_ik)
    # Joint range
    q_mins = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),0]
    q_maxs = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),1]
    # Store MuJoCo state
    if restore_state:
        env.store_state()
    # Initial IK pose
    if q_init is not None:
        env.forward(q=q_init,joint_idxs=joint_idxs_fwd,increase_tick=False)
    # Initialize IK information
    ik_info = init_ik_info()
    add_ik_info(
        ik_info  = ik_info,
        body_name= body_name_trgt,
        p_trgt   = p_trgt,
        R_trgt   = R_trgt, 
    )
    # Loop
    q_curr = env.get_qpos_joints(joint_names=joint_names_for_ik)
    for ik_tick in range(max_ik_tick):
        dq,ik_err_stack = get_dq_from_ik_info(
            env            = env,
            ik_info        = ik_info,
            stepsize       = ik_stepsize,
            eps            = ik_eps,
            th             = ik_th,
            joint_idxs_jac = joint_idxs_jac,
        )
        q_curr = q_curr + dq[joint_idxs_jac] # update
        q_curr = np.clip(q_curr,q_mins,q_maxs) # clip
        env.forward(q=q_curr,joint_idxs=joint_idxs_fwd,increase_tick=False) # fk
        ik_err = np.linalg.norm(ik_err_stack) # IK error
        if ik_err < ik_err_th: break # terminate condition
        if verbose:
            print ("[%d/%d] ik_err:[%.3f]"%(ik_tick,max_ik_tick,ik_err))
        if render:
            if ik_tick%render_every==0:
                plot_ik_info(env,ik_info)
                env.render()
    # Print if IK error is too high
    if verbose_warning and ik_err > ik_err_th:
        print ("ik_err:[%.4f] is higher than ik_err_th:[%.4f]."%
               (ik_err,ik_err_th))
        print ("You may want to increase max_ik_tick:[%d]"%
               (max_ik_tick))
    # Restore backuped state
    if restore_state:
        env.restore_state()
    # Close viewer
    if render:
        env.close_viewer()
    # Return
    return q_curr,ik_err_stack,ik_info

def solve_ik_and_interpolate(
        env,
        joint_names_for_ik = None,
        body_name_trgt     = None,
        p_trgt             = None,
        R_trgt             = None,
        max_ik_tick        = 500,
        ik_err_th          = 1e-4,
        restore_state      = True,
        jerk_limit         = d2r(360),
        vel_interp_max     = d2r(90),
        vel_interp_min     = d2r(10),
    ):
    """ 
        Solve IK and interpolate
    """
    # Start joint position
    q_start = env.get_qpos_joints(joint_names=joint_names_for_ik)

    # Solve IK
    q_final,ik_err_stack,ik_info = solve_ik(
        env=env,joint_names_for_ik=joint_names_for_ik,
        body_name_trgt = body_name_trgt,
        p_trgt         = p_trgt,
        R_trgt         = R_trgt,
        max_ik_tick    = max_ik_tick,
        ik_err_th      = ik_err_th,
        restore_state  = restore_state,
    )

    # Interpolate
    q_anchors = np.vstack((q_start,q_final))
    times,traj_interp,traj_smt,times_anchor = interpolate_and_smooth_nd(
        anchors        = q_anchors,
        HZ             = env.HZ,
        x_lowers       = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),0], 
        x_uppers       = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),1],
        jerk_limit     = jerk_limit,
        vel_interp_max = vel_interp_max,
        vel_interp_min = vel_interp_min,
    )

    # Return
    return times,traj_smt

def solve_ik_list_and_interpolate(
        env,
        joint_names_for_ik = None,
        body_name_trgt     = None,
        p_trgt_list        = [],
        R_trgt_list        = [],
        max_ik_tick        = 500,
        ik_err_th          = 1e-4,
        restore_state      = True,
        jerk_limit         = d2r(360),
        vel_interp_max     = d2r(90),
        vel_interp_min     = d2r(10),
    ):
    """ 
        Solve IK and interpolate
    """
    # Start joint position
    q_start = env.get_qpos_joints(joint_names=joint_names_for_ik)

    # Solve IK
    q_anchors = []
    q_anchors.append(q_start)
    for (p_trgt,R_trgt) in zip(p_trgt_list,R_trgt_list):
        q_final,ik_err_stack,ik_info = solve_ik(
            env=env,joint_names_for_ik=joint_names_for_ik,
            body_name_trgt = body_name_trgt,
            p_trgt         = p_trgt,
            R_trgt         = R_trgt,
            max_ik_tick    = max_ik_tick,
            ik_err_th      = ik_err_th,
            restore_state  = restore_state,
        )
        q_anchors.append(q_final)

    # Interpolate
    q_anchors = np.vstack(q_anchors)
    times,traj_interp,traj_smt,times_anchor = interpolate_and_smooth_nd(
        anchors        = q_anchors,
        HZ             = env.HZ,
        x_lowers       = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),0], 
        x_uppers       = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),1],
        jerk_limit     = jerk_limit,
        vel_interp_max = vel_interp_max,
        vel_interp_min = vel_interp_min,
    )

    # Return
    return times,traj_smt

def animate_chains_slider(env,secs,chains):
    """ 
        Animate chains with slider
    """
    # Reset
    env.reset(step=True)
    # Initialize slider
    L = len(secs)
    sliders = MultiSliderClass(
        n_slider      = 2,
        title         = 'Slider Tick',
        window_width  = 900,
        window_height = 100,
        x_offset      = 100,
        y_offset      = 100,
        slider_width  = 600,
        label_texts   = ['tick','mode (0:play,1:slider,2:reverse)'],
        slider_mins   = [0,0],
        slider_maxs   = [L-1,2],
        slider_vals   = [0,1.0],
        resolutions   = [0.1,1.0],
        verbose       = False,
    )
    # Loop
    env.init_viewer(transparent=True)
    tick,mode = 0,'slider' # 'play' / 'slider'
    while env.is_viewer_alive():
        # Update
        env.increase_tick()
        sliders.update() # update slider
        chain = chains[tick]
        sec = secs[tick]

        # Mode change
        if sliders.get_slider_values()[1] == 0.0: mode = 'play'
        elif sliders.get_slider_values()[1] == 1.0: mode = 'slider'
        elif sliders.get_slider_values()[1] == 2.0: mode = 'reverse'

        # Render
        if env.loop_every(tick_every=20) or (mode=='play') or (mode=='reverse'):
            chain.plot_chain_mujoco(
                env,
                r_link            = 0.02,
                rgba_link         = (0.5, 0.5, 0.98, 0.5),
                plot_joint        = True,
                plot_joint_axis   = True,
                plot_joint_sphere = False,
                plot_joint_name   = False,
                axis_len_joint    = 0.05,
                axis_width_joint  = 0.01,
                plot_rev_axis     = True,
            )
            env.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
            # env.plot_time(p=np.array([0,0,1]),post_str=' mode:[%s]'%(mode))
            env.plot_text(
                p     = np.array([0,0,1]),
                label = '[%d] tick:[%d] time:[%.2f]sec mode:[%s]'%(env.tick,tick,sec,mode)
            )
            env.render()        

        # Proceed
        if mode == 'play':
            if tick < len(chains)-1: tick = tick + 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
        elif mode == 'slider':
            tick = int(sliders.get_slider_values()[0])
        elif mode == 'reverse':
            if tick > 0: tick = tick - 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            
    # Close viewer and slider
    env.close_viewer() 
    sliders.close()
