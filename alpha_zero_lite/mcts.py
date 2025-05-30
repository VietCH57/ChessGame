import numpy as np
import torch
import torch.nn.functional as F
import traceback
from chess_board import Position, PieceType, PieceColor, BOARD_SIZE
from alpha_zero_lite.board_converter import board_to_input, move_to_policy_idx

class MCTS:
    """
    Monte Carlo Tree Search implementation for AlphaZero.
    """
    def __init__(self, network, num_simulations=100, exploration_factor=1.0, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.network = network
        self.num_simulations = num_simulations
        self.exploration_factor = exploration_factor
        self.device = device
        self.tree = {}
    
    def search(self, board):
        """
        Run MCTS search simulations and return the root node
        """
        try:
            # Reset tree for a new search
            self.tree = {}
            
            # Initialize the root node first with board evaluation
            root_hash = self._get_board_hash(board)
            self._create_node(board)
            
            # Check if root node was created successfully
            if root_hash not in self.tree:
                print("Failed to create root node")
                return None
                
            # Run simulations
            for i in range(self.num_simulations):
                try:
                    self._simulate(board.copy_board())
                except Exception as e:
                    print(f"Simulation {i} failed: {e}")
                    traceback.print_exc()
                    continue
            
            # Return the root node
            return self.tree.get(root_hash)
            
        except Exception as e:
            print(f"MCTS search failed: {e}")
            traceback.print_exc()
            return None
    
    def _create_node(self, board):
        """Create a new node in the tree"""
        board_hash = self._get_board_hash(board)
        
        # Skip if already in tree
        if board_hash in self.tree:
            return self.tree[board_hash]
        
        try:
            # Get policy and value from neural network
            state = board_to_input(board, self.device)
            with torch.no_grad():
                policy_logits, value = self.network(state)
            
            # Get valid moves and create mask
            valid_moves = self._get_all_valid_moves_with_indices(board, board.turn)
            mask = np.zeros(BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            for move, idx in valid_moves:
                mask[idx] = 1.0
            
            policy = F.softmax(policy_logits.squeeze(), dim=0).cpu().numpy() * mask
            
            # Normalize if there are valid moves
            if np.sum(policy) > 0:
                policy = policy / np.sum(policy)
            
            # Create new node
            new_node = {
                'visit_count': 0,
                'value_sum': 0,
                'mean_value': 0,
                'prior': 1.0,  # Root node has prior 1.0
                'children': {},
                'children_data': [],
                'is_expanded': False,
                'board': board,
                'to_play': board.turn,
                'policy': policy,
                'raw_value': value.item()
            }
            
            self.tree[board_hash] = new_node
            return new_node
            
        except Exception as e:
            print(f"Error creating node: {e}")
            traceback.print_exc()
            return None
    
    def _simulate(self, board):
        """
        Run a single MCTS simulation from the root to a leaf
        """
        node_path = []
        current_board = board
        current_hash = self._get_board_hash(current_board)
        
        # Make sure root node is in the tree
        if current_hash not in self.tree:
            self._create_node(current_board)
            
        if current_hash not in self.tree:
            print("Root node creation failed in simulation")
            return
        
        # Selection phase: Go down the tree following best moves until a leaf node
        while current_hash in self.tree and self.tree[current_hash]['is_expanded']:
            node = self.tree[current_hash]
            node_path.append(node)
            
            # If no children, break
            if not node['children']:
                break
                
            # Select action with highest UCB score
            best_move = None
            best_score = float('-inf')
            
            for move_tuple, child_hash in node['children'].items():
                child = self.tree.get(child_hash)
                if child is None:
                    continue
                    
                # PUCT formula from AlphaZero paper
                U = self.exploration_factor * child['prior'] * np.sqrt(node['visit_count']) / (1 + child['visit_count'])
                Q = child['mean_value']
                score = Q + U
                
                if score > best_score:
                    best_score = score
                    best_move = move_tuple
            
            if best_move is None:
                # No valid move found
                break
            
            # Apply the selected move
            from_pos, to_pos = Position(*best_move[0]), Position(*best_move[1])
            move_result = current_board.move_piece(from_pos, to_pos)
            if move_result is None:
                # Invalid move
                break
                
            current_hash = self._get_board_hash(current_board)
        
        # Expansion phase: If the node is not expanded, expand it
        if current_hash in self.tree and not self.tree[current_hash]['is_expanded']:
            leaf_node = self.tree[current_hash]
            node_path.append(leaf_node)
            
            # Expand by adding all possible children
            valid_moves = self._get_all_valid_moves_with_indices(leaf_node['board'], leaf_node['to_play'])
            
            for (from_pos, to_pos), move_idx in valid_moves:
                # Create a new board with the move applied
                next_board = leaf_node['board'].copy_board()
                move_result = next_board.move_piece(from_pos, to_pos)
                if move_result is None:
                    continue
                    
                next_hash = self._get_board_hash(next_board)
                
                # Add child to tree if not already there
                if next_hash not in self.tree:
                    child_node = self._create_node(next_board)
                    if child_node is None:
                        continue
                        
                    child_node['prior'] = leaf_node['policy'][move_idx]
                
                # Add to parent's children
                move_key = ((from_pos.row, from_pos.col), (to_pos.row, to_pos.col))
                leaf_node['children'][move_key] = next_hash
                
                leaf_node['children_data'].append({
                    'move': (from_pos, to_pos),
                    'prior': leaf_node['policy'][move_idx] if move_idx < len(leaf_node['policy']) else 0.0,
                    'visits': 0,
                    'value': 0,
                    'hash': next_hash
                })
            
            leaf_node['is_expanded'] = True
        else:
            # If current_hash is not in tree, create it now
            if current_hash not in self.tree:
                new_node = self._create_node(current_board)
                if new_node:
                    node_path.append(new_node)
                    leaf_node = new_node
                else:
                    return
            else:
                leaf_node = self.tree[current_hash]
                if leaf_node not in node_path:
                    node_path.append(leaf_node)
        
        # Backup: update statistics up the search path
        value = leaf_node['raw_value']
        
        for node in reversed(node_path):
            node['visit_count'] += 1
            node['value_sum'] += value if node['to_play'] == leaf_node['to_play'] else -value
            node['mean_value'] = node['value_sum'] / node['visit_count']
            
            # Update children data
            for child_data in node['children_data']:
                child_hash = child_data['hash']
                if child_hash in self.tree:
                    child = self.tree[child_hash]
                    child_data['visits'] = child['visit_count']
                    child_data['value'] = child['mean_value']
    
    def _get_board_hash(self, board):
        """
        Create a hash representing the board state
        """
        try:
            # Use the board's own hash function
            return board.get_board_hash()
        except Exception as e:
            # Fallback hash method if the board's hash function fails
            print(f"Board hash function failed: {e}. Using fallback hash")
            board_state = ""
            for row in range(8):
                for col in range(8):
                    piece = board.get_piece(Position(row, col))
                    board_state += f"{piece.type.value}_{piece.color.value}_"
            board_state += f"turn_{board.turn.value}"
            return hash(board_state)
    
    def _get_all_valid_moves_with_indices(self, board, color):
        """
        Get all valid moves with their policy indices using the simpler encoding
        """
        moves_with_indices = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece and piece.color == color:
                    valid_moves = board.get_valid_moves(pos)
                    for move in valid_moves:
                        # Calculate policy index with the simpler encoding
                        policy_idx = move_to_policy_idx(move.start_pos, move.end_pos)
                        moves_with_indices.append(((move.start_pos, move.end_pos), policy_idx))
        
        return moves_with_indices