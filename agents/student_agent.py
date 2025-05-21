# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.tree = StudentAgent.MCTree(self)
    # stores whether our agent is player 1 (1) or player 2 (2)
    self.player = 0

  class MCTree:
    # exploration constant
    c = np.sqrt(2)

    def __init__(self, parent_agent):
      self.parent_agent = parent_agent
      # root of the search tree starts as None
      self.root = None

      # we will store a hash map of all the (relative) depth 2 nodes in the tree because they could possibly be the next
      # node that step is called on so we can easily retrieve it and make it the new root
      self.potential_next_roots = {}
      # depth of the current root (used to determine next potential roots)
      self.root_depth = 0

    def find_or_create_root(self, board_state):
      key = hash(board_state.tobytes())
      if key in self.potential_next_roots:
        #self.parent_agent.num_existing_roots += 1
        new_root = self.potential_next_roots[key]
        self.root_depth = new_root.depth
        new_root.parent = None
        self.potential_next_roots.clear()

        # build new hashmap of existing depth 2 nodes (potential next roots)
        for child in new_root.children:
          for grandchild in child.children:
            self.potential_next_roots[hash(grandchild.board.tobytes())] = grandchild
        return new_root
      else:
        # if root didn't exist, clearly no grandchildren exist so reset map and relative root depth
        self.potential_next_roots.clear()
        self.root_depth = 0
        return StudentAgent.MCTree.Node(board_state, None, None, self.parent_agent.player, 0)

    # select node to expand
    def select(self):
      current = self.root
      cur_player = self.parent_agent.player

      while True:
        num_valid_moves = len(get_valid_moves(current.board, cur_player))
        # if node is terminal, can't iterate further
        if num_valid_moves == 0:
          return current
        # if node is not fully expanded (has children not in the tree)
        elif num_valid_moves > len(current.children):
          # return the new child added to the tree
          return self.expand(current)
        # otherwise continue search on the best child using UCB
        else:
          current = max(current.children, key=lambda child: self.upper_confidence_formula(child))
          # flip to opponent's turn
          cur_player = 3 - cur_player

    def expand(self, node):
      explored_moves = {child.move for child in node.children}
      for move in get_valid_moves(node.board, node.cur_player):
        if move not in explored_moves:
          resulting_state = deepcopy(node.board)
          execute_move(resulting_state, move, node.cur_player)
          # new node will be the opposite player's turn
          new = StudentAgent.MCTree.Node(resulting_state, node, move, 3-node.cur_player, node.depth + 1)
          # if at relative depth 2, add to potential_next_roots
          if new.depth == (self.root_depth + 2):
            self.potential_next_roots[hash(new.board.tobytes())] = new
          node.children.append(new)
          return new

    @staticmethod
    def rollout(node):
      result = 1
      winner = 0
      cur_board = deepcopy(node.board)

      # when first called on a child node it will be the other player's move
      cur_player = node.cur_player
      is_end, p1_score, p2_score = check_endgame(cur_board, cur_player, 3-cur_player)
      while not is_end:
        move = random_move(cur_board, cur_player)
        if move is not None:
          execute_move(cur_board, move, cur_player)
        # opponent's move
        cur_player = 3 - cur_player
        is_end, p1_score, p2_score = check_endgame(cur_board, cur_player, 3 - cur_player)

      # regardless of which player we are, draw result is the same
      if p1_score == p2_score:
        # keep winner as 0 to indicate a draw
        result = 0.5
      elif p1_score > p2_score:
        winner = 1
      elif p2_score > p1_score:
        winner = 2

      # return the winner and the reward
      return winner, result

    # in this function, we update each node's Q(s,a) also
    # call this on the result of select
    @staticmethod
    def backpropagate(node, winner, result):
      cur_node = node
      while cur_node is not None:
        cur_node.num_sims += 1
        # if they won or was a draw, add to their reward sum
        # note it is flipped because the node's cur_player represents who's turn it is next,
        # not the player whose turn led to that move
        if winner == (3 - cur_node.cur_player) or winner == 0:
          cur_node.reward_sum += result
          cur_node.q_value = cur_node.reward_sum / cur_node.num_sims
        cur_node = cur_node.parent

    # do an iteration of MCTS
    def explore(self):
      selected = self.select()
      # call rollout on the result of select because select will either return a terminal node or the result of
      # expanding the selected node
      winner, result = self.rollout(selected)
      self.backpropagate(selected, winner, result)

    @staticmethod
    def upper_confidence_formula(node):
      return node.q_value + StudentAgent.MCTree.c * np.sqrt(np.log(node.parent.num_sims) / node.num_sims)

    class Node:
      def __init__(self, board, parent, move, cur_player, depth):
        self.board = board
        self.parent = parent
        self.cur_player = cur_player # which player's turn it is to make the next move
        self.depth = depth
        self.move = move # the move that led to this state
        self.children = []
        self.num_sims = 0
        self.reward_sum = 0
        self.q_value = 0

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    start_time = time.time()
    self.player = player

    # get new root of the tree
    self.tree.root = StudentAgent.MCTree.find_or_create_root(self.tree, chess_board)

    # build the tree until near the time limit
    while time.time() - start_time < 1.89:
        self.tree.explore()

    # find the best child of the root (best move)
    best_move = max(self.tree.root.children, key=lambda child: child.q_value).move
    return best_move
