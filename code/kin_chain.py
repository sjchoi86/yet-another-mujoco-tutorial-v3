import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_edges
from util import r2rpy,rpy2r,rodrigues

class KinematicChainClass(object):
    def __init__(self,name='Kinematic Chain'):
        """
            Initialize Kinematic Chain Object
        """
        self.name        = name 
        self.chain       = None
        self.joint_names = []
        # Initialize chain
        self.init_chain()
        
    def init_chain(self):
        """
            Initialize chain
        """
        if self.chain is not None:
            self.chain.clear()
        self.chain = nx.DiGraph(name=self.name)
        
    def get_n_joint(self):
        """
            Get the number of joints
        """
        return self.chain.number_of_nodes()
    
    def get_joint_idx(self,joint_name):
        """
            Get the index of a joint
        """
        joint_idx = self.joint_names.index(joint_name)
        return joint_idx
    
    def get_joint_idxs(self,joint_names):
        """
            Get the indices of joints 
        """
        joint_idxs = [[idx for idx,item in enumerate(self.joint_names) 
                       if item==joint_name] for joint_name in joint_names]
        return joint_idxs
    
    def set_joint_q(self,joint_names,qs):
        """ 
            Set joint values
        """
        for (joint_name,q) in zip(joint_names,qs):
            self.chain.nodes[self.get_joint_idx(joint_name)]['q'] = q

    def add_joint(self,name='',a=np.array([0,0,0]),
                  p=np.zeros(3),R=np.eye(3),
                  p_offset=np.zeros(3),R_offset=np.eye(3),
                  parent_name=None):
        """
            Add joint to the chain
        """
        # Add new node (=joint)
        new_joint_idx = self.get_n_joint()
        self.chain.add_node(new_joint_idx)
        
        # Update joint information
        joint_info = {'name':name,'p':p,'R':R,'q':0.0,
                      'a':a,'p_offset':p_offset,'R_offset':R_offset,
                      'parent':[],'childs':[]}
        self.chain.update(nodes=[(new_joint_idx,joint_info)])
        
        # Append joint name
        self.joint_names.append(name)
        
        # Add parent 
        if parent_name is not None:
            # Add parent index
            parent_idx = self.get_joint_idx(parent_name)
            self.chain.nodes[new_joint_idx]['parent'] = parent_idx
            # Connect parent and child
            self.chain.add_edge(u_of_edge=parent_idx,v_of_edge=new_joint_idx)
        
        # Append childs to the parent
        parent_idx = self.get_joint_idx(name)
        parent_childs = self.chain.nodes[parent_idx]['childs']
        parent_childs.append(new_joint_idx)
        
    def get_joint(self,joint_idx):
        """
            Get joint in tree
        """
        joint = self.chain.nodes[joint_idx]
        return joint
    
    def update_joint_info(self,joint_idx,key,value):
        """
            Update joint information 
        """
        self.chain.nodes[joint_idx][key] = value
        
    def forward_kinematics(self):
        """
            Forward Kinematics
        """
        for idx,edge in enumerate(dfs_edges(self.chain,source=0)):
            idx_fr   = edge[0]
            idx_to   = edge[1]
            joint_fr = self.get_joint(idx_fr)
            joint_to = self.get_joint(idx_to)
            # Update p
            p = joint_fr['R']@joint_to['p_offset'] + joint_fr['p']
            self.update_joint_info(idx_to,'p',p)
            # Update R
            a_to = joint_to['a']
            if abs(np.linalg.norm(a_to)-1) < 1e-6: # with axis
                q_to = joint_to['q']
                R = joint_fr['R']@joint_to['R_offset']@rodrigues(a=a_to,q_rad=q_to)
            else:
                R = joint_fr['R']@joint_to['R_offset']
            self.update_joint_info(idx_to,'R',R)
    
    
    def print_chain_info(self):
        """
            Print chain information
        """
        n_joint = self.get_n_joint()
        for j_idx in range(n_joint):
            joint = self.get_joint(joint_idx=j_idx)
            print ("[%d/%d] joint name:[%s] p:%s rpy_deg:%s"%
                   (j_idx,n_joint,joint['name'],
                    joint['p'],
                    np.degrees(r2rpy(joint['R']))
                   ))
        
    def plot_chain_graph(self,align='horizontal',figsize=(6,4),
                   node_size=300,font_size_node=10,node_colors=None,
                   font_size_title=10,ROOT_ON_TOP=True):
        """
            Plot chain graph
        """
        n_joint = self.get_n_joint()
        tree = self.chain
        for layer, nodes in enumerate(nx.topological_generations(tree)):
            for node in nodes:
                tree.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(tree,align=align,
                                     scale=1.0,subset_key='layer')
        # Invert the tree so that the root node comes on the top
        if ROOT_ON_TOP:
            pos = {node: (x, -y) for node, (x, y) in pos.items()} 
        # Plot
        fig,ax = plt.subplots(figsize=figsize)
        if node_colors is None:
            node_colors = []
            for j_idx in range(n_joint):
                a = self.get_joint(j_idx)['a']
                if np.linalg.norm(a) < 1e-6:
                    node_color = (1,1,1,0.5)
                else:
                    node_color = [0,0,0,0.5]
                    node_color[np.argmax(a)] = 1
                    node_color = tuple(node_color)
                node_colors.append(node_color)
        nx.draw_networkx(tree,pos=pos,ax=ax,with_labels=True,
                         node_size=node_size,font_size=font_size_node,
                         node_color=node_colors,
                         linewidths=1,edgecolors='k')
        ax.set_title('%s'%(tree.name),fontsize=font_size_title)
        fig.tight_layout()
        plt.show()
        
    def plot_chain_mujoco(
        self,env,
        r_link=0.005,rgba_link=(0,0,0,0.5),
        axis_len_joint=0.05,axis_width_joint=0.005,r_joint=0.01,rgba_joint=(0.1,0.1,0.1,0.9),
        r_axis = 0.015,h_axis=0.1
        ):
        """ 
            Plot chain in MuJoCo
        """
        
        # Plot link
        for idx,edge in enumerate(dfs_edges(self.chain,source=0)):
            joint_fr = self.get_joint(edge[0])
            joint_to = self.get_joint(edge[1])
            env.plot_cylinder_fr2to(p_fr=joint_fr['p'],p_to=joint_to['p'],
                                    r=r_link,rgba=rgba_link)
            
        # Plot joint
        for j_idx in range(self.get_n_joint()):
            joint = self.get_joint(j_idx)
            env.plot_T(p=joint['p'],R=joint['R'],
                       PLOT_AXIS=True,axis_len=axis_len_joint,axis_width=axis_width_joint,
                       PLOT_SPHERE=True,sphere_r=r_joint,sphere_rgba=rgba_joint,
                       label=joint['name'])
        
        # Plot revolute axis
        for j_idx in range(self.get_n_joint()):
            joint = self.get_joint(j_idx)
            a = joint['a']
            if np.linalg.norm(a) > 1e-6:
                p,R = joint['p'],joint['R']
                p2 = p + R@a*h_axis
                axis_color = [0,0,0,0.5]
                axis_color[np.argmax(a)] = 1
                env.plot_arrow_fr2to(p_fr=p,p_to=p2,r=r_axis,rgba=axis_color)
        
print ("Ready.")        