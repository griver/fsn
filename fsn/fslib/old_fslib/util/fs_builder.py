from ..fs.deimpl import MotorFS, AdditiveMotorFS
from ..fs.deimpl import SimpleMotivFS, MotivationFS, SecondaryFS
from ..fs import BaseFSNetwork

from . import de_equations as de_eqs
from . import lm_equations as lm_eqs
from ..fs.lmimpl import LMSecondary, OneActivationSecondary

from ..env import Environment

#-----action_fs----------------------------------
act_x0 = 0.25 #0.25
act_k = 10.0
act_edge = 1.5
act_tau = 1.0
act_alpha = 1.0#0.3
act_beta = 2.0
act_gamma = 2.0
act_x1 = 0.1
act_tauc0 = 1 #0.01
act_tauc1 = 25
act_sigma1 = 0.001 #0.001
act_sigma2 = 0.001 #0.001
#----/action_fs-----------------------------------
#-----secondary_fs--------------------------------
sec_x0 = 0.25
sec_k = 10
sec_edge = 1.0
sec_tau = 1.0
sec_alpha = 1.5 # 1.0
sec_beta = 2.0
sec_gamma = 2.0
sec_x1 = 0.1
sec_tauc0 = 1.0 #0.01
sec_tauc1 = 30.0
sec_sigma1 = 0.001#0.001
sec_sigma2 = 0.001#0.001
#----/secondary_fs--------------------------------
#-----motivational_fs--------------------------------
motiv_edge = 2.0
mot_k = 10
mot_x0 = 0.5
mot_sigma1 = 0.01
mot_sigma2 = 0.01
mot_alpha = 1.0 # 1.0
mot_beta = 2.0
mot_gamma = 0.0
mot_edge = 2.0

mot_tau = 1.0
mot_tau1 = 3 # 20
mot_tau2 = 30
mot_tau3 = 10
mot_tau4 = 10
#----/motivational_fs--------------------------------


def create_empty_network(name = "FSNetwork"):
    return BaseFSNetwork(act_edge, sec_edge, motiv_edge, name)


def create_network(env, create_motor, name = "FSNetwork"):
    assert isinstance(env, Environment)

    net = BaseFSNetwork(act_edge, sec_edge, motiv_edge, name)
    for i in xrange(env.get_dimension()):
        mot0 = create_motor(env, net, i, 0)
        mot1 = create_motor(env, net, i, 1)
        net.add_motor(mot0)
        net.add_motor(mot1)

    return net


# when motor_fs is connected to edge index instead of vertex coordinate
def create_edge_moving_network(env, create_motor, name = "FSNetwork"):
    assert isinstance(env, Environment)
    net = BaseFSNetwork(act_edge, sec_edge, motiv_edge, name)

    max_degree = max(len(v.get_outcoming()) for v in env.vertices())

    for i in xrange(max_degree):
        mot0 = create_motor(env, net, i)
        net.add_motor(mot0)

    return net




def simple_motiv(env, goal):
    return SimpleMotivFS(env, goal)


def motiv(env, goal):
    delta_si = de_eqs.delta_si_maker(mot_k, mot_x0, mot_sigma1)
    delta_ri = de_eqs.delta_ri_maker(mot_tau1, mot_sigma2)
    calc_I = de_eqs.ii_maker(mot_alpha, mot_beta, mot_gamma, 0)

    calc_IA = lambda x: 1.0
    calc_AR = de_eqs.mot_ar_maker( mot_tau1, mot_tau)
    numerical_integration = de_eqs.delta_RS_with_noise(de_eqs.delta_RS_rk4(1.0))
    m = MotivationFS(env, de_eqs.delta_RS_rk4(), delta_si, delta_ri, calc_IA, calc_AR, calc_I, goal)
    return m

def secondary(env, motiv_fs, prev, goal):
    delta_si = de_eqs.delta_si_maker(sec_k, sec_x0, sec_sigma1)
    delta_ri = de_eqs.delta_ri_maker(sec_tau, sec_sigma2)
    delta_ci = de_eqs.delta_ci_maker(sec_k, sec_x1, sec_tauc0, sec_tauc1)
    calc_IA = de_eqs.sec_ia_maker(sec_k,0.1)
    calc_AR = de_eqs.sec_ar_maker(sec_k,0.1)
    #calc_IA = eqs.foo_maker(21, 31)
    #calc_AR = eqs.foo_maker(81, 91)
    calc_ii = de_eqs.ii_maker(sec_alpha, sec_beta, sec_gamma, 0)
    sec =  SecondaryFS(env, delta_si, delta_ri, delta_ci, calc_IA, calc_AR, calc_ii, motiv_fs, prev, goal)
    #sec.active_threshold = 2.0
    return sec

def motor(env, net, index, coord_val):
    assert  isinstance(net, BaseFSNetwork)
    motiv_cn = net.get_cnet(net.MOTIV_NET)
    delta_si = de_eqs.delta_si_maker(act_k, act_x0, act_sigma1)
    delta_ri = de_eqs.delta_ri_maker(act_tau, act_sigma2)
    delta_ci = de_eqs.delta_ci_maker(act_k, act_x1, act_tauc0, act_tauc1)
    #calc_IA = de_eqs.motor_ia_maker(act_k, 0.1, coord_val) #eqs.foo_maker(0, 240)  #
    calc_IA = de_eqs.motor_ia_maker3(de_eqs.discrete_f_maker(0.5), index, coord_val)
    calc_AR = de_eqs.motor_ar_maker(act_k, 0.1, coord_val)
    calc_ii = de_eqs.ii_maker(act_alpha, act_beta, act_gamma, 1)
    return MotorFS(env, delta_si, delta_ri, delta_ci, calc_IA, calc_AR, calc_ii, motiv_cn, index, coord_val)

def edge_moving_motor(env, net, index, coord_val=None):
    assert isinstance(net, BaseFSNetwork)
    motiv_cn = net.get_cnet(net.MOTIV_NET)
    delta_si = de_eqs.delta_si_maker(act_k, act_x0, act_sigma1)
    delta_ri = de_eqs.delta_ri_maker(act_tau, act_sigma2)
    delta_ci = de_eqs.delta_ci_maker(act_k, act_x1, act_tauc0, act_tauc1)
    calc_IA = de_eqs.motor_ia_maker4(de_eqs.discrete_f_maker(0.5), index)
    calc_AR = lambda fs: 0
    calc_ii = de_eqs.ii_maker(act_alpha, act_beta, act_gamma, 1)
    return MotorFS(env, delta_si, delta_ri, delta_ci, calc_IA, calc_AR, calc_ii, motiv_cn, index, coord_val)



def base_fs(env, motiv_fs, index, coord_val):
    delta_si = de_eqs.delta_si_maker(act_k, act_x0, act_sigma1)
    delta_ri = de_eqs.delta_ri_maker(act_tau, act_sigma2)
    delta_ci = lambda si, ci: 0
    calc_IA = de_eqs.foo_maker(21, 161)
    calc_AR = de_eqs.foo_maker(150, 161)
    calc_ii = de_eqs.ii_maker(act_alpha, 1.6, act_gamma, 1) #act_beta
    m =  MotorFS(env, delta_si, delta_ri, delta_ci, calc_IA, calc_AR, calc_ii, motiv_fs, index, coord_val)
    m.deactivation_method = lambda: True
    return m


#deactivation in case of absence of the environment reaction
def lm_secondary(net, env, motiv_fs, prev, goal):
    calc_IA = lm_eqs.sec_IA_maker()
    calc_AR = lm_eqs.sec_AR_maker()
    calc_ii = lm_eqs.sec_ii_maker(0.5)
    return LMSecondary(net, env, motiv_fs, prev, goal, calc_IA, calc_AR, calc_ii, lm_eqs.delta_si)


#deactivation after one activity period
def lm_secondary2(net, env, motiv_fs, prev, goal, deactivation_delay=20):
    calc_IA = lm_eqs.weighted_IA_maker(1.0) #lm_eqs.sec_IA_maker()
    calc_AR = lm_eqs.sec_AR_maker()
    calc_ii = lm_eqs.sec_ii_maker(0.3)
    return OneActivationSecondary(net, env, motiv_fs, prev, goal,
                                  calc_IA, calc_AR, calc_ii, lm_eqs.delta_si_with_deactivation,
                                  activity_limit=deactivation_delay)


#deactivation_disabled
def lm_secondary3(net, env, motiv_fs, prev, goal):
    calc_IA = lm_eqs.sec_IA_maker()

    calc_IA = de_eqs.foo_maker(25, 65)
    calc_AR = lm_eqs.sec_AR_maker()
    calc_ii = lm_eqs.sec_ii_maker(0.5)

    lm=LMSecondary(net, env, motiv_fs, prev, goal, calc_IA, calc_AR, calc_ii, lm_eqs.delta_si)
    lm.deactivation_method = MotorFS.deactivation_method
    return lm