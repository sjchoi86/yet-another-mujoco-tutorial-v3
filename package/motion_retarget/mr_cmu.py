import numpy as np
""" 
    Assume that the main notebook called 'sys.path.append('../../package/helper/')'
"""
from transformation import pr2t

def get_T_joi_from_chain_cmu(chain,hip_between_pelvis=True):
    """ 
        Get joints of interest of CMU mocap chain
    """
    p_hip,R_hip = chain.get_joint_pR(joint_name='Hips')
    p_spine,R_spine = chain.get_joint_pR(joint_name='Spine')

    p_rs,R_rs = chain.get_joint_pR(joint_name='RightArm')
    p_re,R_re = chain.get_joint_pR(joint_name='RightForeArm')
    p_rw,R_rw = chain.get_joint_pR(joint_name='RightHand')

    p_ls,R_ls = chain.get_joint_pR(joint_name='LeftArm')
    p_le,R_le = chain.get_joint_pR(joint_name='LeftForeArm')
    p_lw,R_lw = chain.get_joint_pR(joint_name='LeftHand')

    p_neck,R_neck = chain.get_joint_pR(joint_name='Neck')
    p_neck_z = 0.5*(p_rs+p_ls) # z neck position to be the center of two shoulder positions
    p_neck = np.array([p_neck[0],p_neck[1],p_neck_z[2]])

    p_head,R_head = chain.get_joint_pR(joint_name='Head')

    p_rp,R_rp = chain.get_joint_pR(joint_name='RightUpLeg')
    p_rk,R_rk = chain.get_joint_pR(joint_name='RightLeg')
    p_ra,R_ra = chain.get_joint_pR(joint_name='RightFoot')

    p_lp,R_lp = chain.get_joint_pR(joint_name='LeftUpLeg')
    p_lk,R_lk = chain.get_joint_pR(joint_name='LeftLeg')
    p_la,R_la = chain.get_joint_pR(joint_name='LeftFoot')

    p_r_toe,R_r_toe = chain.get_joint_pR(joint_name='RightToeBase')
    p_l_toe,R_l_toe = chain.get_joint_pR(joint_name='LeftToeBase')

    # Modify Hip positions
    
    if hip_between_pelvis:
        p_hip = 0.5*(p_rp+p_lp)
        p_hip_z = 0.5*(p_rp+p_lp) # z hip position to be the center of two pelvis positions
        p_hip = np.array([p_hip[0],p_hip[1],p_hip_z[2]])
    else:
        p_hip = p_hip

    T_joi = {
        'hip': pr2t(p_hip,R_hip),
        'spine': pr2t(p_spine,R_spine),
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),
        'neck':pr2t(p_neck,R_neck),
        'head':pr2t(p_head,R_head),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'rtoe': pr2t(p_r_toe,R_r_toe),
        'ltoe': pr2t(p_l_toe,R_l_toe),
    }
    return T_joi

