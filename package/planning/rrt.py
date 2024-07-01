import numpy as np
import networkx as nx # handle tree
import matplotlib as mpl
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_edges

class RapidlyExploringRandomTreesStarClass(object):
    """
        Rapidly-Exploring Random Trees (RRT) Class
    """
    def __init__(self,name,point_min=np.array([-1,-1]),point_max=np.array([+1,+1]),
                 goal_select_rate=0.1,steer_len_max=0.1,norm_ord=2,search_radius=0.3,
                 n_node_max=10000,TERMINATE_WHEN_GOAL_REACHED=False,
                 SPEED_UP=True):
        """
            Initialize RRT object
        """
        self.name             = name
        self.point_min        = point_min
        self.point_max        = point_max
        self.point_root       = None
        self.point_goal       = None
        self.dim              = len(self.point_min)
        self.goal_select_rate = goal_select_rate
        self.steer_len_max    = steer_len_max # maximum steer length
        self.norm_ord         = norm_ord      # norm order (2, inf, ... )
        self.search_radius    = search_radius
        self.tree             = None
        self.loop_cnt         = 0
        self.n_node_max       = n_node_max
        self.TERMINATE_WHEN_GOAL_REACHED = TERMINATE_WHEN_GOAL_REACHED
        # Speed-up computation by storing points and costs 
        self.SPEED_UP         = SPEED_UP
        self.point_data       = np.zeros((self.n_node_max+1,self.dim))
        self.cost_data        = np.zeros(self.n_node_max+1)
        
    def init_rrt_star(self,point_root=None,point_goal=None,seed=0):
        """
            Initialize RRT*
        """
        # Fix random seed
        np.random.seed(seed=seed)
        # Clear tree
        if self.tree is not None:
            self.tree.clear()
        if point_root is None:
            self.point_root = np.zeros(self.dim)
        else:
            self.point_root = point_root
        # Init tree
        self.tree = tree = nx.DiGraph(name=self.name) # directed graph
        self.add_node(point=self.point_root,cost=0.0,node_parent=None)

        # Set goal point
        if point_goal is not None: self.point_goal = point_goal
        # Initialize loop count
        self.loop_cnt = 0

    def increase_loop_cnt(self):
        """
            Increase loop counter
        """
        self.loop_cnt = self.loop_cnt + 1
        return True
        
    def set_goal(self,point_goal):
        """
            Set Goal
        """
        self.point_goal = point_goal
        
    def add_node(self,point=None,cost=None,node_parent=None):
        """
            Add node to tree
        """
        node_new = self.get_n_node()
        if node_new > self.n_node_max:
            print ("[add_node] node_new:[%d] exceeds n_node_max:[%d]"%
                   (node_new,self.n_node_max))
        self.tree.add_node(node_new)
        if point is not None:
            self.tree.update(
                nodes=[(node_new,{'point':point})]
            )
        if cost is not None:
            self.tree.update(
                nodes=[(node_new,{'cost':cost})]
            )
        if node_parent is not None:
            self.tree.add_edge(node_parent,node_new)
        # Store point and cost
        if point is not None: self.point_data[node_new,:] = point
        if cost is not None: self.cost_data[node_new] = cost
        return node_new
            
    def get_n_node(self):
        """
            Get number of nodes
        """
        return self.tree.number_of_nodes()
    
    def get_nodes(self):
        """ 
            Get tree nodes
        """
        return self.tree.nodes
    
    def get_node_info(self,node):
        """
            Get tree node information
        """
        return self.tree.nodes[node]
    
    def update_node_info(self,node,point=None,cost=None):
        """
            Update node information
        """
        if point is not None:
            self.tree.nodes[node]['point'] = point
        if cost is not None:
            self.tree.nodes[node]['cost'] = cost
    
    def get_edges(self):
        """ 
            Get tree edges
        """
        return self.tree.edges
        
    def get_node_nearest(self,point):
        """
            Get nearest node
        """
        if self.SPEED_UP:
            point_data = self.point_data[:self.get_n_node(),:]
            distances = np.sqrt(np.sum((point_data-point)**2,axis=1))
        else:
            distances = [
                self.get_dist_to_node(node=node,point=point)
                for node in self.tree.nodes
            ]
        node_nearest = np.argmin(distances)
        return node_nearest
    
    def get_node_point(self,node):
        """
            Get node point
        """
        return self.tree.nodes[node]['point'].copy()
    
    def get_node_cost(self,node):
        """
            Get node cost
        """
        return self.tree.nodes[node]['cost']
    
    def get_cost_goal(self):
        """
            Get the cost to goal
        """
        node_goal = self.get_node_goal()
        if node_goal is not None: 
            cost_goal = self.get_node_cost(node_goal)
        else: 
            cost_goal = np.inf
        return cost_goal
    
    def get_node_point_and_cost(self,node):
        """
            Get node point and cost
        """
        point = self.get_node_point(node)
        cost = self.get_node_cost(node)
        return point,cost
    
    def get_dist(self,point1,point2):
        """
            Get distance
        """
        return np.linalg.norm(point1-point2,ord=self.norm_ord)
    
    def get_dist_to_node(self,node,point):
        """
            Get distance from node to point
        """
        return self.get_dist(self.tree.nodes[node]['point'],point)
    
    def get_dist_to_goal(self):
        """
            Get distance from tree to goal
        """
        node_nearest = self.get_node_nearest(self.point_goal)
        dist_goal = self.get_dist_to_node(node=node_nearest,point=self.point_goal)
        return dist_goal
    
    def get_node_goal(self,eps=1e-6):
        """
            Get goal node
        """
        node_nearest = self.get_node_nearest(self.point_goal)
        dist_goal = self.get_dist_to_node(node=node_nearest,point=self.point_goal)
        if dist_goal < eps:
            node_goal = node_nearest
        else:
            node_goal = None
        return node_goal
    
    def get_node_parent(self,node):
        """
            Get parent node
        """
        node_parent = [node for node in self.tree.predecessors(node)][0]
        return node_parent
    
    def get_path_to_goal(self):
        """
            Get path to goal
        """
        node_goal = self.get_node_goal()
        if node_goal is None: # RRT not finished yet
            path_to_goal = None
            node_list = []
            return path_to_goal,node_list
        path_list = [self.point_goal]
        node_list = [node_goal]
        parent_node = [node for node in self.tree.predecessors(node_goal)][0]
        while parent_node:
            path_list.append(self.tree.nodes[parent_node]['point'])
            node_list.append(parent_node)
            parent_node = [node for node in self.tree.predecessors(parent_node)][0]
        path_list.append(self.point_root)
        node_list.append(0)
        path_list.reverse() # start from root, end with goal
        node_list.reverse()
        path_to_goal = np.array(path_list) # [L x D]

        # Update cost information of 'path_node_list'
        cost_sum = 0
        for idx,node in enumerate(node_list):
            point_curr = self.get_node_info(node)['point']
            if idx > 0:
                node_parent = self.get_node_parent(node)
                point_parent = self.get_node_info(node_parent)['point']
                cost_sum = cost_sum + self.get_dist(point_curr,point_parent)
                # Update cost
                self.update_node_info(node,point=None,cost=cost_sum)

        return path_to_goal,node_list
    
    def sample_point(self):
        """
            Sample point
        """
        point_range = self.point_max-self.point_min
        point_rand = self.point_min+point_range*np.random.rand(self.dim)
        return point_rand
    
    def steer(self,node_nearest,point_sample):
        """
            Steer
        """
        # Find the nearest point in the tree
        point_nearest = self.get_node_point(node=node_nearest)
        
        vector = point_sample - point_nearest
        length = np.linalg.norm(vector)
        if length == 0:
            # If the tree already contains 'point_sample', skip this turn
            point_steer,cost_steer = None,None
        else:
            stepsize = min(self.steer_len_max,length)
            point_steer = point_nearest + vector/np.linalg.norm(vector)*stepsize
            cost_nearest = self.get_node_cost(node=node_nearest)
            cost_steer = cost_nearest + \
                self.get_dist_to_node(node=node_nearest,point=point_steer)
        return point_steer,cost_steer
    
    def get_nodes_near(self,point,search_radius=None):
        """
            Get the list of nodes near 'point' w.r.t given 'search_radius'
        """
        # Get distances of all nodes to 'node'
        if self.SPEED_UP:
            point_data = self.point_data[:self.get_n_node(),:]
            distances = np.sqrt(np.sum((point_data-point)**2,axis=1))
        else:
            distances = [
                self.get_dist(self.get_node_point(node),point)
                for node in self.get_nodes()
            ]
        # Accumulate the list of near nodes thresholded by 'search_radius'
        if search_radius is None:
            search_radius = self.search_radius
        if self.SPEED_UP:
            nodes_near = np.where(distances<search_radius)[0]
        else:
            nodes_near = []
            for node,dist in enumerate(distances):
                if dist <= search_radius:
                    nodes_near.append(node)
        return nodes_near

    def replace_node_parent(self,node,node_parent_new):
        """
            Rewire 'node' from 'node_parent_curr' to 'node_parent_new'
        """
        # Remove current parent
        node_parent = self.get_node_parent(node)
        self.tree.remove_edge(node_parent,node)
        # Connect new parent
        self.tree.add_edge(node_parent_new,node)

    def update_nodes_cost(self,node_source=0,VERBOSE=False):
        """
            Update the costs of all nodes
        """
        for edge in dfs_edges(self.tree,source=node_source): # for all edges is DFS
            node_parent,node_child = edge[0],edge[1]
            point_parent,cost_parent = self.get_node_point_and_cost(node_parent)
            point_child,cost_child = self.get_node_point_and_cost(node_child)
            # Update child cost
            cost_child_new = cost_parent+self.get_dist(point_parent,point_child)
            if VERBOSE:
                if cost_parent != cost_child_new:
                    print ("[update_nodes_cost] node_child:[%d] cost_child:[%.2f]=>[%.2f]"%
                        (node_child,cost_parent,cost_child_new))
            self.update_node_info(node_child,cost=cost_child_new)

    def plot_tree(self,figsize=(6,6),nodesize=50,arrowsize=10,linewidth=1,
                  nodecolor='w',edgecolor='k',xlim=(-1,+1),ylim=(-1,+1),
                  title_str=None,titlefs=10,SKIP_PLT_SHOW=False):
        """
            Plot tree
        """
        if self.dim == 2:
            pos = {node:self.tree.nodes[node]['point'] for node in self.tree.nodes}
        else:
            pos = nx.spring_layout(self.tree,seed=0)
        plt.figure(figsize=figsize)
        ax = plt.axes()
        nx.draw_networkx_nodes(
            self.tree,pos=pos,node_size=nodesize,node_color=nodecolor,
            linewidths=linewidth,edgecolors=edgecolor,ax=ax)
        nx.draw_networkx_edges(
            self.tree,pos=pos,node_size=nodesize,edge_color=edgecolor,
            width=linewidth,arrowstyle="->",arrowsize=arrowsize,ax=ax)
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set(xlim=xlim,ylim=ylim)
        if title_str is None:
            title_str = "Tree of [%s]"%(self.name)
        plt.title(title_str,fontsize=titlefs)
        if not SKIP_PLT_SHOW:
            plt.show()

    def plot_tree_custom(self,
                         figsize=(6,6),xlim=(-1.01,1.01),ylim=(-1.01,1.01),
                         nodems=3,nodemec='k',nodemfc='w',nodemew=1/2,
                         edgergba=[0,0,0,0.2],edgelw=1/2,
                         startrgb=[1,0,0],startms=8,startmew=2,startfs=10,startbbalpha=0.5,start_xoffset=0.1,
                         goalrgb=[0,0,1],goalms=8,goalmew=2,goalfs=10,goalbbalpha=0.5,goal_xoffset=0.1,
                         pathrgba=[1,0,1,0.5],pathlw=5,pathtextfs=8,
                         obs_list=[],obsrgba=[0.2,0.2,0.2,0.5],obsec='k',
                         textfs=8,titlestr=None,titlefs=12,
                         PLOT_PATH_TEXT=False,PLOT_FULL_TEXT=False,
                         SAVE_PNG=False,png_path='',VERBOSE=True,
                         ):
        """
            Plot tree without using networkx package
        """
        plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        # Get node positions
        pos = np.array([self.get_node_point(node) for node in self.get_nodes()])
        # Plot edges
        edgelist = list(self.get_edges())
        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
        edge_collection = mpl.collections.LineCollection(
            edge_pos,colors=edgergba[:3],linewidths=edgelw,alpha=edgergba[3])
        ax.add_collection(edge_collection)
        # Plot nodes
        plt.plot(pos[:,0],pos[:,1],'o',ms=nodems,mfc=nodemfc,mec=nodemec,mew=nodemew)
        # Plot obstacles
        colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,len(obs_list))])
        colors[:,3] = 0.5
        for obs_idx,obs in enumerate(obs_list):
            if obsrgba is None:
                color = colors[obs_idx]
            else:
                color = obsrgba
            plt.fill(*obs.exterior.xy,fc=color,ec=obsec)
        # Path to goal
        path_to_goal,path_node_list = self.get_path_to_goal()
        if path_to_goal is not None:
            plt.plot(path_to_goal[:,0],path_to_goal[:,1],'o-',
                    color=pathrgba,lw=pathlw,mec='k',mfc='none')
        for node_idx in range(self.get_n_node()):
            node = self.get_nodes()[node_idx]
            if PLOT_FULL_TEXT:
                plt.text(node['point'][0],node['point'][1],'  [%d] %.2f'%(node_idx,node['cost']),
                        color='k',fontsize=textfs,va='center')
        # Root to goal path
        for idx,node_idx in enumerate(path_node_list):
            node = self.get_nodes()[node_idx]
            if (idx > 0) and (idx < len(path_node_list)):
                if PLOT_PATH_TEXT:
                    plt.text(node['point'][0],node['point'][1],'  [%d] %.2f'%(node_idx,node['cost']),
                            color='k',fontsize=pathtextfs,va='center')
        # Start position
        plt.plot(self.point_root[0],self.point_root[1],'o',
                mfc='none',mec=startrgb,ms=startms,mew=startmew)
        plt.text(self.point_root[0]+start_xoffset,self.point_root[1],'Start',
                color=startrgb,fontsize=startfs,va='center',
                bbox=dict(fc='white',alpha=startbbalpha,ec='none'))
        # Goal position
        plt.plot(self.point_goal[0],self.point_goal[1],'o',
                mfc='none',mec=goalrgb,ms=goalms,mew=goalmew)
        plt.text(self.point_goal[0]+goal_xoffset,self.point_goal[1],'Goal',
                color=goalrgb,fontsize=goalfs,va='center',
                bbox=dict(fc='white',alpha=goalbbalpha,ec='none'))
        # Axes
        plt.xticks(fontsize=8); plt.yticks(fontsize=8)
        # Axis again
        plt.axis([xlim[0],xlim[1],ylim[0],ylim[1]])
        ax.set_aspect('equal', adjustable='box')
        # Title
        if titlestr is None: titlestr = '%s'%(self.name)
        plt.title(titlestr,fontsize=titlefs)
        if SAVE_PNG:
            if VERBOSE:
                print ("[%s] saved."%(png_path))
            plt.savefig(png_path, bbox_inches='tight')
            plt.close()
        else:
            # Show
            plt.show()

def is_qpos_feasible(
        env,
        qpos,
        joint_names,
        robot_body_names,
        obj_body_names,
        env_body_names,
    ):
    """
        Collsion checker (using FK)
        Following conditions will be considered as infeasible
        1. robot_body_names-robot_body_names
        2. robot_body_names-obj_body_names
        3. robot_body_names-env_body_names
    """
    # backup state
    qpos_backup = env.get_qpos()
    # FK and collision check
    env.forward(q=qpos,joint_names=joint_names)
    if_feasible = True
    for c_idx in range(env.data.ncon):
        contact = env.data.contact[c_idx]
        b1 = env.body_names[env.model.geom_bodyid[contact.geom1]]
        b2 = env.body_names[env.model.geom_bodyid[contact.geom2]]
        if (b1 in robot_body_names) and (b2 in robot_body_names):
            if_feasible = False
            break
        if (b1 in robot_body_names) and (b2 in obj_body_names):
            if_feasible = False
            break
        if (b1 in obj_body_names) and (b2 in robot_body_names):
            if_feasible = False
            break
        if (b1 in robot_body_names) and (b2 in env_body_names):
            if_feasible = False
            break
        if (b1 in env_body_names) and (b2 in robot_body_names):
            if_feasible = False
            break
    # restore state
    env.forward(q=qpos_backup)
    # return
    return if_feasible

def is_qpos_connectable(
        env,
        qpos1,
        qpos2,
        joint_names,
        robot_body_names,
        obj_body_names,
        env_body_names,
        deg_th = 5.0,
    ):
    """
        Collsion checker (using FK)
        Following conditions will be considered as infeasible
        1. robot_body_names-robot_body_names
        2. robot_body_names-obj_body_names
        3. robot_body_names-env_body_names
    """
    # Interpolate
    q_dist_deg = np.rad2deg(np.linalg.norm(qpos2-qpos1,ord=np.inf))
    n = 2+int(q_dist_deg/deg_th)
    q_interp = np.linspace(start=qpos1,stop=qpos2,num=n)
    # Check each interpolated points
    if_connectable = True
    for idx in range(n):
        q_check = q_interp[idx,:]
        is_feasible = is_qpos_feasible(
            env,
            q_check,
            joint_names,
            robot_body_names,
            obj_body_names,
            env_body_names,
        )
        if not is_feasible:
            if_connectable = False
            break
    # return
    return if_connectable