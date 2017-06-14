# coding=utf-8
from ..env.environment import Environment
from ..env.point import Point
from ..graph.edge import Edge
from ..stochasticenv import StochasticEnvironment
from ..stochasticenv.stochastic_environment import random_array
import numpy as np
import networkx as nx

#--------Environments-------------------------------------------------------
def line():
    env = Environment(1, "line")
    p0 = Point((0,))
    p1 = Point((1,))
    env.add_vertex(p0)
    env.add_vertex(p1)
    env.add_edge(p0, p1, 1)
    env.set_current_state(p0)
    return env


def chain(n):
    env = Environment(1, "chain")
    p0 = Point((0,))
    p1 = Point((1,))
    env.add_vertex(p0)
    env.add_vertex(p1)
    env.add_edge(p0, p1, 1)
    env.set_current_state(p0)
    return env


def square():
    env = direct_square()
    v = env.vertices()
    env.add_edge(v[1], v[0], 1)
    env.add_edge(v[2], v[0], 1)
    env.add_edge(v[3], v[1], 1)
    env.add_edge(v[3], v[2], 1)
    return env


def direct_square():
    env = Environment(2, "square")
    p00 = Point((0, 0))
    p01 = Point((0, 1))
    p10 = Point((1, 0))
    p11 = Point((1, 1))

    env.add_vertex(p00)
    env.add_vertex(p01)
    env.add_vertex(p10)
    env.add_vertex(p11)

    env.add_edge(p00, p01, 1)
    env.add_edge(p00, p10, 1)
    env.add_edge(p01, p11, 1)
    env.add_edge(p10, p11, 1)
    env.set_current_state(p00)
    return env


def cube():
    env = Environment(3, "cube")
    points = []
    for x in [0,1]:
        for y in [0, 1]:
            for z in [0, 1]:
                p = Point((x, y, z))
                env.add_vertex(p)
                points.append(p)

    env.add_edge(points[0], points[1], 1)
    env.add_edge(points[0], points[4], 1)
    env.add_edge(points[0], points[2], 1)
    env.add_edge(points[2], points[6], 1)
    env.add_edge(points[2], points[3], 1)
    env.add_edge(points[3], points[7], 1)

    env.add_edge(points[1], points[0], 1)
    env.add_edge(points[4], points[0], 1)
    env.add_edge(points[2], points[0], 1)
    env.add_edge(points[6], points[2], 1)
    env.add_edge(points[3], points[2], 1)
    env.add_edge(points[7], points[3], 1)

    env.set_current_state(points[0])
    return env


def slingshot():
    env = Environment(3, "slingshot")

    points = []
    points.append(Point((0, 0, 0)))  # 0
    points.append(Point((0, 1, 0)))  # 1
    points.append(Point((1, 1, 0)))  # 2
    points.append(Point((0, 1, 1)))  # 3
    points.append(Point((1, 0, 0)))  # 4
    points.append(Point((0, 0, 1)))  # 5
    for p in points:
        env.add_vertex(p)

    env.add_edge(points[0], points[1], 1)
    env.add_edge(points[1], points[0], 1)
    env.add_edge(points[1], points[2], 1)
    env.add_edge(points[2], points[1], 1)
    env.add_edge(points[2], points[4], 1)
    env.add_edge(points[4], points[2], 1)
    env.add_edge(points[1], points[3], 1)
    env.add_edge(points[3], points[1], 1)
    env.add_edge(points[3], points[5], 1)
    env.add_edge(points[5], points[3], 1)

    env.set_current_state(points[0])
    return env
#--------/Environments-------------------------------------------------------


#--------Stochastic Environments-------------------------------------------------------
def fork():
    env = StochasticEnvironment(2, "Fork" , 0)
    points = [Point((0, 0)), Point((0, 1)), Point((1, 0))]

    for p in points:
        env.add_vertex(p)
    e1 = env.add_edge(points[0], points[1])
    e2 = env.add_edge(points[0], points[2])
    env.create_competetive_group([e1, e2])
    return env, e1, e2

def rhomb():
    env = StochasticEnvironment(3, "Rhomb" , 0)
    points = [Point((0, 0, 0)), # 0
              Point((0, 1, 0)), # 1
              Point((1, 1, 0)), # 2
              Point((0, 1, 1)), # 3
              Point((1, 1, 1))] # 4

    for p in points:
        env.add_vertex(p)
    env.add_edge(points[0], points[1])
    env.add_edge(points[1], points[2])
    env.add_edge(points[1], points[3])
    e1 = env.add_edge(points[2], points[4])
    e2 = env.add_edge(points[3], points[4])

    env.add_edge(points[1], points[0])
    env.add_edge(points[2], points[1])
    env.add_edge(points[3], points[1])

    env.create_competitive_group([e1, e2], (0.7, 0.3))
    return env

def binary_choice(left_prob):
    env = StochasticEnvironment(5, "BinaryChoice", 0)
    points = [Point((0, 0, 0, 0, 0)),  # 0
              Point((1, 0, 0, 0, 0)),  # 1
              Point((0, 0, 0, 0, 1)),  # 2
              Point((1, 1, 0, 0, 0)),  # 3
              Point((0, 0, 0, 1, 1)),  # 4
              Point((1, 1, 1, 0, 0)),  # 5
              Point((0, 0, 1, 1, 1)),  # 6
              Point((1, 1, 1, 1, 0)),  # 7
              Point((0, 1, 1, 1, 1)),  # 8
              Point((1, 1, 1, 1, 1))]  # 9

    for p in points:
        env.add_vertex(p)

    env.add_both_edges(points[0], points[1])
    env.add_both_edges(points[1], points[3])
    env.add_both_edges(points[3], points[5])
    env.add_both_edges(points[5], points[7])

    env.add_both_edges(points[0], points[2])
    env.add_both_edges(points[2], points[4])
    env.add_both_edges(points[4], points[6])
    env.add_both_edges(points[6], points[8])

    e1 = env.add_edge(points[7], points[9])
    e2 = env.add_edge(points[8], points[9])

    env.create_competitive_group([e1, e2], [left_prob, 1.0 - left_prob])
    return env

def stochastic_env1():
    env = StochasticEnvironment(4, "StochasticEnv1", 0)
    points = [Point((0, 0, 0, 0)),  # 0
              Point((0, 0, 0, 1)),  # 1
              Point((0, 1, 0, 1)),  # 2
              Point((0, 1, 0, 0)),  # 3
              Point((0, 1, 1, 0)),  # 4
              Point((0, 0, 1, 0)),  # 5
              Point((0, 0, 1, 1)),  # 6
              Point((0, 1, 1, 1)),  # 7
              Point((1, 1, 1, 1)),  # 8
              Point((1, 0, 1, 1)),  # 9
              Point((1, 1, 1, 0)),  # 10
              Point((1, 0, 1, 0))]  # 11

    for p in points:
        env.add_vertex(p)

    env.add_both_edges(points[0], points[1])  # 0 <-> 1 -> 2 <-> 7
    e1 = env.add_edge(points[1], points[2])
    env.add_both_edges(points[2], points[7])

    env.add_both_edges(points[0], points[3])  # 0 <-> 3 <-> 4 -> 7
    env.add_both_edges(points[3], points[4])
    e2 = env.add_edge(points[4], points[7])

    e3 = env.add_edge(points[0], points[5])  # 0 -> 5 <-> 6 <-> 7
    env.add_both_edges(points[5], points[6])
    env.add_both_edges(points[6], points[7])

    env.add_both_edges(points[7], points[8])  # 7 <-> 8

    env.add_both_edges(points[8], points[10])  # 8 <-> 10 -> 11
    e4 = env.add_edge(points[10], points[11])

    e5 = env.add_edge(points[8], points[9])  # 8 -> 9 <-> 11
    env.add_both_edges(points[9], points[11])

    env.create_competetive_group([e1, e2, e3])
    env.create_competetive_group([e4, e5])
    return env


def stochastic_env2(left_prob):
    env = StochasticEnvironment(2, "StochasticEnv1", 0)
    points = [Point((0, 0)),  # 0
              Point((0, 1)),  # 1
              Point((1, 0)),  # 2
              Point((1, 1))]  # 3

    for p in points:
        env.add_vertex(p)

    e1 = env.add_edge(points[0], points[1])
    env.add_edge(points[1], points[3])

    e2 = env.add_edge(points[0], points[2])
    env.add_edge(points[2], points[3])

    env.create_competetive_group([e1, e2], [left_prob, 1.0 - left_prob])
    return env

def grid33():
    env = StochasticEnvironment(4, "StochasticEnv1", 1)
    points = [Point((1, 0, 0, 0)),  # 0
              Point((0, 0, 0, 0)),  # 1
              Point((0, 0, 0, 1)),  # 2
              Point((1, 1, 0, 0)),  # 3
              Point((0, 1, 0, 0)),  # 4
              Point((0, 1, 0, 1)),  # 5
              Point((1, 1, 1, 0)),  # 6
              Point((0, 1, 1, 0)),  # 7
              Point((0, 1, 1, 1))]  # 8

    for p in points:
        env.add_vertex(p)

    env.add_both_edges(points[0], points[1])
    env.add_both_edges(points[1], points[2])
    e1 = env.add_edge(points[0], points[3])
    e2 = env.add_edge(points[1], points[4])
    e3 = env.add_edge(points[2], points[5])

    env.add_both_edges(points[3], points[4])
    env.add_both_edges(points[4], points[5])
    e4 = env.add_edge(points[3], points[6])
    e5 = env.add_edge(points[4], points[7])
    e6 = env.add_edge(points[5], points[8])

    env.add_both_edges(points[6], points[7])
    env.add_both_edges(points[7], points[8])

    return env, [e1,e2, e3], [e4, e5, e6]

def torus(x, y, env_class=None, get_probabilities=None):
    if env_class is None:
        env_class = StochasticEnvironment

    size = int(np.ceil(np.log2(x*y)))
    binary =  '{0:0' + str(size) + 'b}'
    env = env_class(size, start_state_id=0)
    points = np.empty((x, y), dtype=Point)
    edges = []

    for i in xrange(x):
        for j in xrange(y):
            points[i][j] = Point(tuple(binary.format(i*y + j)))
            env.add_vertex(points[i][j])

    for i in xrange(x):
        for j in xrange(y):
            e1e2 = env.add_both_edges(points[i][j], points[i][(j + 1) % y])  # right
            edges.append(e1e2)
            e1e2 = env.add_both_edges(points[i][j], points[(i + 1) % x][j])  # bottom
            edges.append(e1e2)

    if get_probabilities is None:
        return env

    probs = get_probabilities(len(edges))  # число пар ребер в два раза больше числа вершин

    for i in xrange(len(edges)):
        env.create_united_group(edges[i], probs[i])
    return env


def sparse_grid(h, w, get_probabilities=None, env_class=StochasticEnvironment, cell_len=1):
    """
    Создает 'разреженный' тороидальный граф. Под разреженным имеется ввиду, то что сторона
    каждой ячейки прямоугольной решетки теперь может иметь длинну в несколько ребер вместо одного.
    Например при cell_len = 3, h = 1, w = 2 граф будет выглядет так:
     |     |     |
    -o--o--o--o--o--o-
     |     |     |
    -o     o     o
     |     |     |
    -o--o--o--o--o--o-
     |     |     |
     o     o     o

    :param H:  высота прямоугольной решетки(в ячейках).
    :param W:  ширина прямоугольной решетки(в ячейках).
    :param env_class: класс определяюший тип среды(в основном особенности стохастического изменения ребер)
    :param get_probabilites: вероятности того что переход по стороне ячейки будет доступен
    :param cell_len: количество ребер на стороне ячейки в решетке.
    :return: Возвращает тороидальный граф с заданными параметрами.
    """

    assert cell_len > 0, 'cell_len must be positive integer!'

    h_full_grid = h*cell_len + 1
    w_full_grid = w*cell_len + 1

    n_states_full_grid = h_full_grid * w_full_grid
    n_states = n_states_full_grid - h*w*(cell_len-1)**2
    #print 'num_states:', n_states
    state_dim = int(np.ceil(np.log2(n_states)))
    binary_template = '{0:0' + str(state_dim) +'b}'
    env = env_class(state_dim, start_state_id=0)

    def id2tuple(idx):
        return map(int, binary_template.format(idx))

    coord2state = {}
    stoch_edges = []
    for i in xrange(h_full_grid):
        for j in xrange(w_full_grid):
            on_row = i % cell_len == 0
            on_column = j % cell_len == 0

            if on_row or on_column:
                state_idx = len(coord2state)
                state = Point(id2tuple(state_idx))
                coord2state[(i,j)] = state
                env.add_vertex(state)

                edges = []
                if on_row:
                    if j > 0:
                        dst = coord2state[(i,j-1)]
                        e = env.add_both_edges(state,dst)
                        edges.append(e)

                if on_column:
                    if i > 0:
                        dst = coord2state[i-1,j]
                        e = env.add_both_edges(state,dst)
                        edges.append(e)

                if on_column and on_row:
                    stoch_edges.extend(edges)

    assert all(e1.is_available() and e2.is_available() for e1,e2 in stoch_edges), \
        'all edges must be available by default!'
    #last to edges goes to a target state. We leave set them permanently available:
    stoch_edges.pop()
    stoch_edges.pop()

    if get_probabilities is None:
        return env

    probs = get_probabilities(len(stoch_edges))  # число пар ребер в два раза больше числа вершин

    for i in xrange(len(stoch_edges)):
        env.create_united_group(stoch_edges[i], probs[i])
    return env


def hierarchy_test_env(env_class=StochasticEnvironment):
    num_nodes = 30
    dim = int(np.ceil(np.log2(num_nodes)))
    binary_template = '{0:0' + str(dim) + 'b}'

    def id2tuple(idx):
        return tuple(map(int, binary_template.format(idx)))

    env = env_class(dim, 'HierarchyTestEnv', 0)
    nodes = [None]*num_nodes

    for idx in xrange(num_nodes):
        nodes[idx] = Point(id2tuple(idx))
        env.add_vertex(nodes[idx])

    def create_chain_and_return_last_transition(start_idx, *node_indices):
        assert len(node_indices) > 0, "You can't build with only one node."
        prev_idx = start_idx
        transition = None
        for idx in node_indices:
            transition = env.add_both_edges(nodes[prev_idx], nodes[idx])
            prev_idx = idx
        return transition

    # TODO: check that this creates permanent edges
    tr2_8 = create_chain_and_return_last_transition(0, 1, 2, 8) # 0 -- 1 -- 2 -- 8
    tr4_8 = create_chain_and_return_last_transition(0, 3, 4, 8) # 0 -- 3 -- 4 -- 8
    tr7_8 = create_chain_and_return_last_transition(0, 5, 6, 7, 8) # 0 -- 5 -- 6 -- 7 -- 8
    env.create_arbitrary_group((tr2_8, 0.2), (tr4_8, 0.2), (tr7_8, 0.6))

    tr9_12 = create_chain_and_return_last_transition(8, 9, 12) # 8 -- 9 -- 12
    tr10_12 = create_chain_and_return_last_transition(8, 10, 12) # 8 -- 10 -- 12
    tr11_12 = create_chain_and_return_last_transition(8, 11, 12) # 8 -- 11 -- 12
    env.create_arbitrary_group((tr9_12, 0.5), (tr10_12, 0.5))
    env.create_united_group(tr11_12, 0.6)

    tr13_29 = create_chain_and_return_last_transition(12, 13, 29) # 12 -- 13 -- 29
    tr14_29 = create_chain_and_return_last_transition(12, 14, 29) # 12 -- 14 -- 29
    #env.create_arbitrary_group((tr13_29, 0.3), (tr14_29, 0.7))

    tr17_20 = create_chain_and_return_last_transition(0, 15, 16, 17, 20) # 0 -- 15 -- 16 -- 17 -- 20
    tr19_20 = create_chain_and_return_last_transition(15, 18, 19, 20)
    env.create_arbitrary_group((tr17_20, 0.3), (tr19_20, 0.7))

    tr22_25 = create_chain_and_return_last_transition(20, 21, 22, 25)
    tr23_25 = create_chain_and_return_last_transition(20, 23, 25)
    tr24_25 = create_chain_and_return_last_transition(20, 24, 25)
    env.create_arbitrary_group((tr22_25, .5), (tr23_25, .3), (tr24_25, .2))

    tr26_29 = create_chain_and_return_last_transition(25, 26, 29)
    tr28_29 = create_chain_and_return_last_transition(25, 27, 28, 29)
    #env.create_arbitrary_group((tr26_29, 0.2), (tr28_29, 0.8))

    env.create_arbitrary_group((tr13_29, 0.03), (tr14_29, 0.27), #end of the left path prob_sum: 0.3
                               (tr26_29, 0.21), (tr28_29, 0.49)) #end of the right path prob_sum: 0.7


    return env


def scc_sample():

    size = int(np.ceil(np.log2(12)))
    binary = '{0:0' + str(size) + 'b}'
    p = np.empty(12, dtype=Point)

    env = StochasticEnvironment(size, "scc_sample", 0)

    for i in xrange(12):
        p[i] = Point(tuple(binary.format(i)))
        env.add_vertex(p[i])

    env.add_edge(p[0], p[1])

    env.add_edge(p[1], p[2])
    env.add_edge(p[1], p[4])
    env.add_edge(p[1], p[3])

    env.add_edge(p[2], p[5])

    env.add_edge(p[4], p[1])
    env.add_edge(p[4], p[5])
    env.add_edge(p[4], p[6])

    env.add_edge(p[5], p[2])
    env.add_edge(p[5], p[7])

    env.add_edge(p[6], p[7])
    env.add_edge(p[6], p[9])

    env.add_edge(p[7], p[10])

    env.add_edge(p[8], p[6])

    env.add_edge(p[9], p[8])

    env.add_edge(p[10], p[11])

    env.add_edge(p[11], p[9])

    return env


######################################### ERDOS-RENYI MODEL RELATED FUNCTIONS #######################################
#####################################################################################################################
from plots import plt


def compute_edge_presence_probability(nodes_number, edges_number):
    """
    Считаем какую долю состоявляет число ребер от числа всех возможных ребер
    """
    all_possible_edges_number = (nodes_number* (nodes_number - 1)/2)
    return float(edges_number) / (all_possible_edges_number)


def read_nx_graph_from_dot(filename):
    """
    Read networkx graph from filename
    :param filename: name of the dot file with graph info
    :return: networkx graph
    """
    graph = nx.read_dot(filename)
    for e in graph.edges_iter(data=True):
        e[2]['is_available'] = e[2]['is_available'] == 'True'
        e[2]['prob'] = float(e[2]['prob'].strip('"'))
    return graph

def simple_draw_graph(graph, start, end):
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph,pos, nodelist=graph.nodes(),
                           node_color='r', node_size=600, alpha=0.8)
    nx.draw_networkx_nodes(graph,pos, nodelist=[start, end],
                           node_color='yellow', node_size=1000, alpha=0.8)
    nx.draw_networkx_labels(graph, pos)

    available_edges = [e[0:2] for e in graph.edges_iter(data=True)]

    nx.draw_networkx_edges(graph, pos, edgelist=available_edges,
                           width=3,alpha=0.9,edge_color='r')
    plt.show()

def draw_graph(graph, start, end):
        pos = nx.circular_layout(graph)
        nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes(),
                               node_color='r', node_size=600, alpha=0.8)
        nx.draw_networkx_nodes(graph, pos, nodelist=[start, end],
                               node_color='yellow', node_size=1000, alpha=0.8)
        nx.draw_networkx_labels(graph, pos)

        available_edges = [e[0:2] for e in graph.edges_iter(data=True) if e[2]['is_available']]
        not_available_edges = [e[0:2] for e in graph.edges_iter(data=True) if not e[2]['is_available']]
        print "number of edges:", graph.number_of_edges()
        print "is_available:", len(available_edges),
        print "not available:", len(not_available_edges)
        nx.draw_networkx_edges(graph, pos, edgelist=available_edges,
                               width=3,alpha=0.9,edge_color='r')

        nx.draw_networkx_edges(graph, pos, edgelist=not_available_edges,
                               width=2,alpha=0.1,edge_color='b')
        plt.show()


def load_stohastic_environment_from_dot(filename, init_env=None):
    nx_graph = read_nx_graph_from_dot(filename)
    if init_env is None:
        init_env = StochasticEnvironment


    size_of_state_vector = int(np.ceil(np.log2(nx_graph.number_of_nodes())))
    binary = '{0:0' + str(size_of_state_vector) + 'b}'

    env = init_env(size_of_state_vector,  start_state_id=0)
    print "environment type is:", type(env)
    points = np.empty(nx_graph.number_of_nodes(),dtype=Point)

    for i in xrange(nx_graph.number_of_nodes()):
        points[i] = Point(tuple(binary.format(i)))
        env.add_vertex(points[i])

    for node1, node2, edge_data in nx_graph.edges_iter(data=True):
        e1e2 = env.add_both_edges(points[int(node1)], points[int(node2)])
        env.create_united_group(e1e2, edge_data['prob'])


    assert len(env._stoch_groups) == nx_graph.number_of_edges(), \
        "Ошибка, число ребер в nx_graph'e должно равнятся числу пар ребер в среде"
    return env

######################################### ERDOS-RENYI MODEL RELATED FUNCTIONS #######################################
#####################################################################################################################

#--------/Stochastic Environments-------------------------------------------------------
#--------/Stochastic Environments-------------------------------------------------------

