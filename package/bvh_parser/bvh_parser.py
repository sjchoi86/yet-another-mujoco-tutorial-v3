from math import radians, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

"""
    Call the following line on the main.py
    sys.path.append("../helper")
"""
from kinematic_chain import KinematicChainClass
from transformation import (
    rpy2r,
    rpy2r_order,
    t2p,
    t2r,
    pr2t,
    t2pr,
)

ZEROMAT = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                 [0., 0., 0., 0.], [0., 0., 0., 0.]])
IDENTITY = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., 1., 0.], [0., 0., 0., 1.]])

class Node(object):
    """Skeleton hierarchy node."""

    def __init__(self, root=False):
        self.name = None
        self.channels = []
        self.offset = (0, 0, 0)
        self.children = []
        self._is_root = root

    @property
    def is_root(self):
        return self._is_root

    @property
    def is_end_site(self):
        return len(self.children) == 0

class BvhReader(object):
    """BioVision Hierarchical (.bvh) file reader."""

    def __init__(self, filename):

        self.filename = filename
        # A list of unprocessed tokens (strings)
        self._token_list = []
        # The current line number
        self._line_num = 0

        # Root node
        self.root = None
        self._node_stack = []

        # Total number of channels
        self.num_channels = 0

    def on_hierarchy(self, root):
        pass

    def on_motion(self, frames, dt):
        pass

    def on_frame(self, values):
        pass

    def read(self):
        """Read the entire file."""
        with open(self.filename, 'r') as self._file_handle:
            self.read_hierarchy()
            self.on_hierarchy(self.root)
            self.read_motion()

    def read_motion(self):
        """Read the motion samples."""
        # No more tokens (i.e. end of file)? Then just return
        try:
            tok = self.token()
        except StopIteration:
            return

        if tok != "MOTION":
            raise SyntaxError("Syntax error in line %d: 'MOTION' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        # Read the number of frames
        tok = self.token()
        if tok != "Frames:":
            raise SyntaxError("Syntax error in line %d: 'Frames:' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        frames = self.int_token()

        # Read the frame time
        tok = self.token()
        if tok != "Frame":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' "
                              "expected, got '%s' instead"
                              % (self._line_num, tok))
        tok = self.token()
        if tok != "Time:":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' "
                              "expected, got 'Frame %s' instead"
                              % (self._line_num, tok))

        dt = self.float_token()

        self.on_motion(frames, dt)

        # Read the channel values
        for i in range(frames):
            s = self.read_line()
            a = s.split()
            if len(a) != self.num_channels:
                raise SyntaxError("Syntax error in line %d: %d float values "
                                  "expected, got %d instead"
                                  % (self._line_num, self.num_channels,
                                     len(a)))
            values = list(map(lambda x: float(x), a))  # In Python 3 map returns map-object, not a list. Can't slice.
            self.on_frame(values)

    def read_hierarchy(self):
        """Read the skeleton hierarchy."""
        tok = self.token()
        if tok != "HIERARCHY":
            raise SyntaxError("Syntax error in line %d: 'HIERARCHY' expected, "
                              "got '%s' instead" % (self._line_num, tok))
        tok = self.token()
        if tok != "ROOT":
            raise SyntaxError("Syntax error in line %d: 'ROOT' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        self.root = Node(root=True)
        self._node_stack.append(self.root)
        self.read_node()

    def read_node(self):
        """Read the data for a node."""

        # Read the node name (or the word 'Site' if it was a 'End Site' node)
        name = self.token()
        self._node_stack[-1].name = name

        tok = self.token()
        if tok != "{":
            raise SyntaxError("Syntax error in line %d: '{' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        while 1:
            tok = self.token()
            if tok == "OFFSET":
                x = self.float_token()
                y = self.float_token()
                z = self.float_token()
                self._node_stack[-1].offset = (x, y, z)
            elif tok == "CHANNELS":
                n = self.int_token()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                   "Xrotation", "Yrotation", "Zrotation"]:
                        raise SyntaxError("Syntax error in line %d: Invalid "
                                          "channel name: '%s'"
                                          % (self._line_num, tok))
                    channels.append(tok)
                self.num_channels += len(channels)
                self._node_stack[-1].channels = channels
            elif tok == "JOINT":
                node = Node()
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "End":
                node = Node()
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "}":
                if self._node_stack[-1].is_end_site:
                    self._node_stack[-1].name = "End Site"
                self._node_stack.pop()
                break
            else:
                raise SyntaxError("Syntax error in line %d: Unknown "
                                  "keyword '%s'" % (self._line_num, tok))

    def int_token(self):
        """Return the next token which must be an int. """
        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Integer expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def float_token(self):
        """Return the next token which must be a float."""
        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Float expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def token(self):
        """Return the next token."""

        # Are there still some tokens left? then just return the next one
        if self._token_list:
            tok = self._token_list[0]
            self._token_list = self._token_list[1:]
            return tok

        # Read a new line
        s = self.read_line()
        self.create_tokens(s)
        return self.token()

    def read_line(self):
        """Return the next line.

        Empty lines are skipped. If the end of the file has been
        reached, a StopIteration exception is thrown.  The return
        value is the next line containing data (this will never be an
        empty string).
        """
        # Discard any remaining tokens
        self._token_list = []
        # Read the next line
        while 1:
            s = self._file_handle.readline()
            self._line_num += 1
            if s == "":
                raise StopIteration
            return s

    def create_tokens(self, s):
        """Populate the token list from the content of s."""
        s = s.strip()
        a = s.split()
        self._token_list = a


class Joint:

    def __init__(self, name):
        self.name = name
        self.children = []
        self.channels = []  # Set later.  Ordered list of channels: each
        # list entry is one of [XYZ]position, [XYZ]rotation
        self.hasparent = 0  # flag
        self.parent = 0  # joint.addchild() sets this
        self.strans = np.array([0., 0., 0.])  # I think I could just use regular Python arrays.

        # Transformation matrices:
        self.stransmat = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                [0., 0., 0., 0.], [0., 0., 0., 0.]])
        
        self.rot = {}  # self.rot[t] Rotation values at the frame.
        self.trtr = {}  # self.trtr[time]  A premultiplied series of translation and rotation matrices.
        self.worldpos = {}  # Time-based worldspace xyz position of the joint's endpoint.  A list of vec4's

    def info(self):
        """ Prints information about the joint to stdout.
        """
        print("Joint name:", self.name)
        print(" %s is connected to " % self.name,)
        if len(self.children) == 0:
            print("nothing")
        else:
            for child in self.children:
                print("%s " % child.name,)
            print()
        for child in self.children:
            child.info()

    def __str__(self):  # Recursively build up text info
        str2 = self.name + " at strans=" + \
            str(self.strans) + " is connected to "
        # Not sure how well self.strans will work now that self.strans is
        # a numpy "array", no longer a cgkit vec3.
        if len(self.children) == 0:
            str2 = str2 + "nothing\n"
        else:
            for child in self.children:
                str2 = str2 + child.name + " "
            str2 = str2 + "\n"
        str3 = ""
        for child in self.children:
            str3 = str3 + child.__str__()
        str1 = str2 + str3
        return str1

    def addchild(self, childjoint):
        self.children.append(childjoint)
        childjoint.hasparent = 1
        childjoint.parent = self


class ReadBVH(BvhReader):

    def on_hierarchy(self, root):
        #    print("readbvh: onHierarchy invoked"
        self.root = root  # Save root for later use
        self.keyframes = []  # Used later in onFrame

    def on_motion(self, frames, dt):
        # print("readbvh: onMotion invoked.  frames = %s, dt = %s" %
        # (frames,dt)
        self.frames = frames
        self.dt = dt

    def on_frame(self, values):
        #   print("readbvh: onFrame invoked, values =", values
        # Hopefully this gives us a list of lists
        self.keyframes.append(values)

def process_bvhnode(node, parentname='hips'):
    name = node.name
    if (name == "End Site") or (name == "end site"):
        name = parentname + "End"
    
    b1 = Joint(name)
    b1.channels = node.channels
    b1.strans[0] = node.offset[0]
    b1.strans[1] = node.offset[1]
    b1.strans[2] = node.offset[2]

    # Compute static translation matrix from vec3 b1.strans
    # cgkit#  b1.stransmat = b1.stransmat.translation(b1.strans)
    #   b1.stransmat = deepcopy(IDENTITY)
    b1.stransmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                          [0., 0., 1., 0.], [0., 0., 0., 1.]])

    b1.stransmat[0, 3] = b1.strans[0]
    b1.stransmat[1, 3] = b1.strans[1]
    b1.stransmat[2, 3] = b1.strans[2]

    for child in node.children:
        b2 = process_bvhnode(child, name)  # Creates a child joint "b2"
        b1.addchild(b2)
    return b1

class Skeleton:

    def __init__(self, hips, keyframes, frames=0, dt=.033333333, ignore_root_offset=True):
        self.root = hips
        # 9/1/08: we now transfer the large bvh.keyframes data structure to
        # the skeleton because we need to keep this dataset around.
        self.keyframes = keyframes
        self.frames = frames  # Number of frames (caller must set correctly)
        self.dt = dt
        # self.edges = []  # List of list of edges.  self.edges[time][edge#]
        self.edges = {}  # As of 9/1/08 this now runs from 1...N not 0...N-1

        # Precompute hips min and max values in all 3 dimensions.
        # First determine how far into a keyframe we need to look to find the
        # XYZ hip positions
        offset = 0
        for channel in self.root.channels:
            if channel == "Xposition":
                xoffset = offset
            if channel == "Yposition":
                yoffset = offset
            if channel == "Zposition":
                zoffset = offset
            offset += 1
        self.minx = 999999999999
        self.miny = 999999999999
        self.minz = 999999999999
        self.maxx = -999999999999
        self.maxy = -999999999999
        self.maxz = -999999999999
        # We will ignore the static hips OFFSET value by default, since
        # it will not reproduce the correct values for world positions in most cases.
        # I feel it's bad BVH file form to have a non-zero HIPS offset
        # position, but there are definitely files that do this (e.g. MotionBuilder BVH Export).
        if ignore_root_offset:
            self.root.strans[0] = 0.0
            self.root.strans[1] = 0.0
            self.root.strans[2] = 0.0
            self.root.stransmat = IDENTITY
        xcorrect = self.root.strans[0]
        ycorrect = self.root.strans[1]
        zcorrect = self.root.strans[2]

        for keyframe in self.keyframes:
            x = keyframe[xoffset] + xcorrect
            y = keyframe[yoffset] + ycorrect
            z = keyframe[zoffset] + zcorrect
            if x < self.minx:
                self.minx = x
            if x > self.maxx:
                self.maxx = x
            if y < self.miny:
                self.miny = y
            if y > self.maxy:
                self.maxy = y
            if z < self.minz:
                self.minz = z
            if z > self.maxz:
                self.maxz = z

    def __str__(self):
        str1 = "frames = " + str(self.frames) + ", dt = " + str(self.dt) + "\n"
        str1 = str1 + self.root.__str__()
        return str1

    @staticmethod
    def joint_dfs(root):
        """
        Go through root's children and return joints.
        :param root: Starting node.
        :return: Children of root.
        :rtype: list
        """
        nodes = []
        stack = [root]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            for child in cur_node.children:
                stack.insert(0, child)
        return nodes
    
    def get_frames_worldpos(self, n=None):
        """Returns a list of frames, first item in list will be a header
        :param n: If not None, returns specified frame (with header).
        :type n: int
        :rtype: tuple
        """
        joints = self.joint_dfs(self.root)

        frame_data = []
        if n is None:
            for i in range(len(self.keyframes)):
                t = i * self.dt
                single_frame = [t, ]
                for j in joints:
                    single_frame.extend(j.worldpos[t][:3])
                frame_data.append(single_frame)
        else:
            t = n * self.dt
            single_frame = [t, ]
            for j in joints:
                single_frame.extend(j.worldpos[t][:3])
            frame_data.append(single_frame)

        header = ["{}.{}".format(j.name, thing) for j in joints
                  for thing in ("X", "Y", "Z")]
        header = ["Time", ] + header
        return header, frame_data
    
    def get_frames_rotations(self, n=None):
        """Returns a list of frames, first item in list will be a header
        :param n: If not None, returns specified frame (with header).
        :type n: int
        :rtype: tuple
        """
        joints = self.joint_dfs(self.root)

        frame_data = []
        if n is None:
            for i in range(len(self.keyframes)):
                t = i * self.dt
                single_frame = [t, ]
                for j in joints:
                    if j.rot:
                        rot = j.rot[t]
                    else:
                        rot = [0.0, 0.0, 0.0]
                    single_frame.extend(rot)
                frame_data.append(single_frame)
        else:
            t = n * self.dt
            single_frame = [t, ]
            for j in joints:
                if j.rot:
                    rot = j.rot[t]
                else:
                    rot = [0.0, 0.0, 0.0]
                single_frame.extend(rot)
            frame_data.append(single_frame)

        header = ["{}.{}".format(j.name, thing) for j in joints
                  for thing in ("X", "Y", "Z")]
        header = ["Time", ] + header
        return header, frame_data

    def get_frame(self, f):
        """
        Get motion values per joint for frame f.
        :param f: Frame
        :type f: int
        :return: A dictionary of {joint.name: (rotation, world position)} for frame f
        :rtype: dict
        """
        joints = self.joint_dfs(self.root)

        frame_data = dict()
        
        t = f * self.dt
        for j in joints:
            frame_data[j.name] = j.rot[t] if t in j.rot else None, j.worldpos[t][:3]
        return frame_data
    
    def get_offsets(self):
        """
        Get the offsets for each joint in the skeleton.
        :return: Dictionary of {joint.name: offset}.
        :rtype: dict
        """
        joints = self.joint_dfs(self.root)
        offsets = dict()
        for j in joints:
            offsets[j.name] = j.strans
        return offsets
    
    def as_dict(self):
        """
        Get the skeleton topology as dictionary.
        :return: Dictionary of {j.name: j.parent, j.strans, j.rot, type, children}
        :rtype: dict
        """
        joints = self.joint_dfs(self.root)
        joints_dict = {}

        for j in joints:
            if not j.hasparent:
                type = 'root'
            else:
                type = 'joint'
            if j.name[-3:] == "End":
                type = 'end'
            
            if j.rot:
                rot_0 = tuple(j.rot[0])
            else:
                rot_0 = None
                
            joints_dict[j.name] = (j.parent.name if j.hasparent else None,
                                   tuple(j.strans),
                                   rot_0,
                                   type,
                                   [child.name for child in j.children])
        return joints_dict


def process_bvhkeyframe(keyframe, joint, t, DEBUG=0):

    counter = 0
    dotrans = 0

    # We have to build up drotmat one rotation value at a time so that
    # we get the matrix multiplication order correct.
    drotmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                     [0., 0., 1., 0.], [0., 0., 0., 1.]])

    if DEBUG:
        print(" process_bvhkeyframe: doing joint %s, t=%d" % (joint.name, t))
        print(" keyframe has %d elements in it." % (len(keyframe)))

    # Suck in as many values off the front of "keyframe" as we need
    # to populate this joint's channels.  The meanings of the keyvals
    # aren't given in the keyframe itself; their meaning is specified
    # by the channel names.
    has_xrot = False
    has_yrot = False
    has_zrot = False
    for channel in joint.channels:
        keyval = keyframe[counter]
        if channel == "Xposition":
            dotrans = 1
            xpos = keyval
        elif channel == "Yposition":
            dotrans = 1
            ypos = keyval
        elif channel == "Zposition":
            dotrans = 1
            zpos = keyval
        elif channel == "Xrotation":
            has_xrot = True
            xrot = keyval
            theta = radians(xrot)
            mycos = cos(theta)
            mysin = sin(theta)
            drotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[1, 1] = mycos
            drotmat2[1, 2] = -mysin
            drotmat2[2, 1] = mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

        elif channel == "Yrotation":
            has_yrot = True
            yrot = keyval
            theta = radians(yrot)
            mycos = cos(theta)
            mysin = sin(theta)
            drotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[0, 0] = mycos
            drotmat2[0, 2] = mysin
            drotmat2[2, 0] = -mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

        elif channel == "Zrotation":
            has_zrot = True
            zrot = keyval
            theta = radians(zrot)
            mycos = cos(theta)
            mysin = sin(theta)
            drotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[0, 0] = mycos
            drotmat2[0, 1] = -mysin
            drotmat2[1, 0] = mysin
            drotmat2[1, 1] = mycos
            drotmat = np.dot(drotmat, drotmat2)
        else:
            print("Fatal error in process_bvhkeyframe: illegal channel"
                  " name ", channel)
            return(0)
        counter += 1
    # End "for channel..."
    if has_xrot or has_yrot or has_zrot:  # End sites don't have rotations.
        joint.rot[t] = (xrot, yrot, zrot)
    
    if dotrans:  # If we are the hips...
        # Build a translation matrix for this keyframe
        dtransmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
        dtransmat[0, 3] = xpos
        dtransmat[1, 3] = ypos
        dtransmat[2, 3] = zpos

        if DEBUG:
            print("  Joint %s: xpos ypos zpos is %s %s %s" % (joint.name, xpos, ypos, zpos))
        # End of IF dotrans

        if DEBUG:
            print("  Joint %s: xrot yrot zrot is %s %s %s" % (joint.name, xrot, yrot, zrot))

    if joint.hasparent:  # Not hips
        parent_trtr = joint.parent.trtr[t]  # Dictionary-based rewrite
        localtoworld = np.dot(parent_trtr, joint.stransmat)

    else:  # Hips
        # cgkit#    localtoworld = joint.stransmat * dtransmat
        localtoworld = np.dot(joint.stransmat, dtransmat)

    trtr = np.dot(localtoworld, drotmat)

    joint.trtr[t] = trtr  # New dictionary-based approach

    # worldpos = localtoworld * ORIGIN  # worldpos should be a vec4
    worldpos = np.array([localtoworld[0, 3], localtoworld[1, 3],
                      localtoworld[2, 3], localtoworld[3, 3]])
    joint.worldpos[t] = worldpos  # Dictionary-based approach

    if DEBUG:
        print("  Joint %s: here are some matrices" % (joint.name))
        print("   stransmat:")
        print(joint.stransmat)
        if not joint.hasparent:  # if hips
            print("   dtransmat:")
            print(dtransmat)
        print("   drotmat:")
        print(drotmat)
        print("   localtoworld:")
        print(localtoworld)
        print("   trtr:")
        print(trtr)
        print("  worldpos:", worldpos)
        print()

    newkeyframe = keyframe[counter:]  # Slices from counter+1 to end
    for child in joint.children:
        # Here's the recursion call.  Each time we call process_bvhkeyframe,
        # the returned value "newkeyframe" should shrink due to the slicing
        # process
        newkeyframe = process_bvhkeyframe(newkeyframe, child, t, DEBUG=DEBUG)
        if newkeyframe == 0:  # If retval = 0
            print("Passing up fatal error in process_bvhkeyframe")
            return 0
    return newkeyframe


def process_bvhfile(filename, DEBUG=0):
    if DEBUG:
        print("Reading BVH file...",)
    my_bvh = ReadBVH(filename)  # Doesn't actually read the file, just creates
    # a readbvh object and sets up the file for
    # reading in the next line.
    my_bvh.read()  # Reads and parses the file.

    hips = process_bvhnode(my_bvh.root)  # Create joint hierarchy
    if DEBUG:
        print("done")

    if DEBUG:
        print("Building skeleton...",)
    myskeleton = Skeleton(hips, keyframes=my_bvh.keyframes, frames=my_bvh.frames, dt=my_bvh.dt)
    if DEBUG:
        print("done")
    if DEBUG:
        print("skeleton is: ", myskeleton)
    return myskeleton

def get_skeleton_from_bvh(bvh_path):
    """ 
        Get skeleton from bvh file
    """
    # Load
    skeleton = process_bvhfile(bvh_path)
    # FK
    for tick in range(skeleton.frames):
        new_frame = process_bvhkeyframe(
            skeleton.keyframes[tick],
            skeleton.root,
            skeleton.dt*tick,
        )
    return skeleton

def get_chains_from_skeleton(
        skeleton,
        env              = None,
        rpy_order        = [2,1,0],
        p_rate           = 0.1, # positional scale
        plot_chain_graph = True,
        plot_init_chain  = True,
        verbose          = True,
    ):
    """ 
        Get chains from skeleton
    """
    L = skeleton.frames
    skel_pos_info,skel_pos_time_list = skeleton.get_frames_worldpos()
    skel_rot_info,skel_rot_time_list = skeleton.get_frames_rotations()
    secs = np.array(skel_pos_time_list)[:,0] # [L]
    skel_pos_array     = np.array(skel_pos_time_list)[:,1:].reshape(L,-1,3) # [L x n_joint x 3]
    skel_rpy_deg_array = np.array(skel_rot_time_list)[:,1:].reshape(L,-1,3) # [L x n_joint x 3]

    # Initialize kinematic chain
    chain = KinematicChainClass(name='CMU-mocap')
    node = skeleton.root # root node
    chain.add_joint(
        name=node.name,
        p=np.array([0,0,0]),
        R=rpy2r(np.radians([0,0,0])),
    )
    deq = deque()
    while (len(node.children)>0) or (len(deq)>0):
        if len(deq) > 0:
            node = deq.pop()
            chain.add_joint(
                name = node.name, 
                parent_name = node.parent.name,
                p_offset = t2p(np.array(node.stransmat))*p_rate,
                R_offset = t2r(np.array(node.stransmat)),
            )
        for child in node.children:
            deq.append(child)
    
    # Copy init chain
    chain_init = deepcopy(chain)
    
    # Root joint name
    root_name = chain.get_root_name()
        
    # Print joint names
    if verbose:
        print ("[Joint names]")
        print ("root_name:[%s]"%(root_name))
        for j_idx,joint_name in enumerate(chain.joint_names):
            print ("[%02d/%d] joint_name:[%s]"%(j_idx,chain.get_n_joint(),joint_name))

    # Plot chain graph
    if plot_chain_graph:
        chain.plot_chain_graph(
            align='vertical',
            figsize=(5,2),
            node_size=200,
            font_size_node=8,
            node_colors=None,
            font_size_title=10,
            ROOT_ON_TOP=True,
        )
        
    if plot_init_chain:
        env.init_viewer(
            title     = "Kinematic Chain",
            width     = 1200,
            height    = 800,
            hide_menu = True,
        )
        env.set_viewer(
            azimuth       = 177,
            distance      = 4.0,
            elevation     = -28,
            lookat        = [0.0,0.1,0.65],
            transparent   = False,
            contactpoint  = True,
            contactwidth  = 0.05,
            contactheight = 0.05,
            contactrgba   = np.array([1,0,0,1]),
            joint         = True,
            jointlength   = 0.5,
            jointwidth    = 0.1,
            jointrgba     =[0.2,0.6,0.8,0.6],
        )
        env.reset()
        chain.set_root_joint_pR(p=np.array([0,0,0]),R=rpy2r(np.radians([0,0,0])))
        chain.forward_kinematics() # forward kinematics chain
        while env.is_viewer_alive():
            chain.plot_chain_mujoco(
                env,
                r_link            = 0.04,
                rgba_link         = (0.5, 0.5, 0.98, 0.5),
                plot_joint        = True,
                plot_joint_axis   = True,
                plot_joint_sphere = False,
                plot_joint_name   = False,
                axis_len_joint    = 0.075,
                axis_width_joint  = 0.01,
                plot_rev_axis     = True,
            )
            env.plot_T(p=np.zeros(3),R=np.eye(3,3),plot_axis=True,axis_len=1.0,axis_width=0.005)
            env.render()
            # Save image
            if (env.render_tick%10)==0: scene_img = env.grab_image()
        # Close viewer
        env.close_viewer()
        # Plot
        plt.figure(figsize=(5,4));plt.imshow(scene_img)
        plt.title("Initial Chain",fontsize=10);plt.axis('off');plt.show()
        print ("Done.")
        
    # Make chains
    chains = []
    for tick in range(L):
        p_root = p_rate*skel_pos_array[tick,0,:]
        R_offset_list = []
        for joint_idx in range(chain.get_n_joint()):
            if joint_idx == 0: # root 
                rpy_deg = np.array([0,0,0])
                rpy_deg_root = skel_rpy_deg_array[tick,joint_idx,:] # backup root rot
            else:
                rpy_deg = skel_rpy_deg_array[tick,joint_idx,:]
            R_offset_list.append(
                rpy2r_order(np.radians(rpy_deg),order=rpy_order)
            )
        R_root = rpy2r_order(np.radians(rpy_deg_root),order=rpy_order)
        chain.set_root_joint_R(R=R_root)
        chain.set_root_joint_p(p=p_root)
        chain.set_joint_R_offset_list(chain.joint_names,R_offset_list)
        chain.forward_kinematics()
        # Append
        chains.append(deepcopy(chain)) # we have to 'deepcopy'
    
    # Return
    return secs,chains

def get_chains_zup(
        chains,
        T_trans_zup = pr2t(np.array([0,0,0]),rpy2r(np.radians([90,0,0]))),
    ):
    """ 
        Get chains from y-up to z-up
    """
    chains_zup = []
    for tick,chain in enumerate(chains):
        p_root,R_root = chain.get_root_joint_pR()
        T_root = pr2t(p_root,R_root)
        T_root_zup = T_trans_zup @ T_root
        p_root_zup,R_root_zup = t2pr(T_root_zup)
        # FK chain
        chain_zup = deepcopy(chain)
        chain_zup.set_root_joint_pR(p=p_root_zup,R=R_root_zup)
        chain_zup.forward_kinematics()
        chains_zup.append(chain_zup) # we have to 'deepcopy'
    # Return        
    return chains_zup

def get_chains_from_bvh_cmu_mocap(
        bvh_path,
        env              = None,
        rpy_order        = [2,1,0],
        p_rate           = 0.056444,
        zup              = True,
        plot_chain_graph = True,
        plot_init_chain  = False,
        verbose          = True,
    ):
    """ 
        Get chains from bvh file of CMU motion capture data
    """
    # Get skeleton
    skeleton = get_skeleton_from_bvh(bvh_path)

    # Get chains
    secs,chains = get_chains_from_skeleton(
        skeleton         = skeleton,
        env              = env,
        rpy_order        = rpy_order, # cmu:[2,1,0], nc:[2,0,1]
        p_rate           = p_rate, # cmu:0.056444
        plot_chain_graph = plot_chain_graph,
        plot_init_chain  = plot_init_chain,
        verbose          = verbose,
    )

    # Z-up
    if zup:
        chains_zup = get_chains_zup( 
            chains,
            T_trans_zup = pr2t(np.array([0,0,0]),rpy2r(np.radians([90,0,0]))),
        )
        chains = chains_zup

    # Return
    return secs,chains
