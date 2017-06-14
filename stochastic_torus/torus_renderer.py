import sys
import os.path as path

import numpy as np
from gym import utils as gym_utils
from collections import namedtuple # for map legend
from .utils import envs

AnsiLegend = namedtuple("AnsiLegend",['horisontal_e','vertical_e', 'node', 'agent', 'target', 'empty'])

default_legend = AnsiLegend(
  horisontal_e='-',
  vertical_e='|',
  node='O',
  agent='A',
  target='T',
  empty=' '
)


class TorusRenderer(object):
  def __init__(self, torus, legend = default_legend):
    '''
    Initialize a new ANSI map
    Args:
      - legend: an instance of the AnsiLegend class.The legend specifies
        ansi symbols used to render a torus
      - size: a typle of two values that represent size of the torus
    '''
    assert isinstance(legend, AnsiLegend)
    assert isinstance(torus, envs.StochasticEnvironment)

    self._env = torus
    self.legend = legend

    n_states = self._env.get_vertices_number()
    x = y = int(np.sqrt(n_states))
    self.size = (x,y)

    self.map = self._create_map()
    self._coords2edges = self._build_coords_to_edges_dict()

    self.prev_state = None

  # ==== util methods for internal use =====================
  def _create_map(self):
    map_tile = np.asarray([
      self.legend.empty + self.legend.vertical_e, # []
      self.legend.horisontal_e + self.legend.node
    ], dtype='c')
    full_map = np.tile(map_tile, self.size)
    return full_map

  def _build_coords_to_edges_dict(self):
    coords2np = self._edge_coords_to_node_pairs_dict()
    np2e = self._node_pairs_to_edges_dict()
    #np_2_e[np][0] to check availability it is sufficient to pick only one edge
    coords2edges ={k:np2e[np][0] for k, np in coords2np.iteritems()}
    return coords2edges

  def _edge_coords_to_node_pairs_dict(self):
    def sorted_pair(a,b): #same as tuple(sorted([a,b]))
      return (a,b) if a < b else (b,a)

    def neighbour_node(node, shift):
      nx,ny = node[0] + shift[0], node[1] + shift[1]
      return nx % self.size[0], ny % self.size[1]

    left = (0,-1)
    up = (-1, 0)
    shifts = (left, up)
    coords2nodes = {}

    for node_pos in np.ndindex(self.size):
      node_id = self._grid_pos_to_node_id(node_pos)
      node_coord = self._grid_pos_to_coords(node_pos)

      for shift in shifts:
        neighbour = neighbour_node(node_pos, shift)
        neighbour_id = self._grid_pos_to_node_id(neighbour)
        node_pair = sorted_pair(node_id, neighbour_id)

        edge_coord = (node_coord[0] + shift[0], node_coord[1] + shift[1])
        coords2nodes[edge_coord] = node_pair

    return coords2nodes

  def _node_id_to_grid_pos(self, node_id):
    y_side = self.size[1]
    return node_id // y_side, node_id % y_side

  def _grid_pos_to_node_id(self, grid_pos):
    y_side = self.size[1]
    return grid_pos[0]*y_side + grid_pos[1]

  def _grid_pos_to_coords(self, grid_pos):
    return 1 + 2*grid_pos[0], 1 + 2*grid_pos[1]

  def _node_id_to_coords(self, node_id):
    grid_pos = self._node_id_to_grid_pos(node_id)
    return self._grid_pos_to_coords(grid_pos)

  def _node_pairs_to_edges_dict(self):
    pairs2edges = {}
    for n in self._env.vertices():
      for e in n.get_outcoming():
        src, dst = e.get_src().get_id(), e.get_dst().get_id()
        pair = (src, dst) if src < dst else (dst, src)
        pairs2edges.setdefault(pair, []).append(e)
    return pairs2edges
  # ==== /util methods for internal use ====================

  def render(self, outfile, curr_state, target_state):
    out = self.map.copy().tolist()

    ax, ay = self._node_id_to_coords(curr_state.get_id())
    tx, ty = self._node_id_to_coords(target_state.get_id())

    out[ax][ay] = self.legend.agent
    out[tx][ty] = self.legend.target

    for coords, edge in self._coords2edges.iteritems():
      ex, ey = coords
      if not edge.is_available():
        out[ex][ey] = self.legend.empty  # remove desabled edges

    out = [[c.decode('utf-8') for c in line] for line in out]

    agent_color = 'red' if curr_state is self.prev_state else 'yellow'
    out[ax][ay] = gym_utils.colorize(out[ax][ay], agent_color, highlight=True)
    if curr_state is not target_state:
      out[tx][ty] = gym_utils.colorize(out[tx][ty], 'green', highlight=True)

    out = [[c + ' ' for c in line] for line in out]
    outfile.write("\n".join(["".join(row) for row in out])+"\n")

    self.prev_state = curr_state
