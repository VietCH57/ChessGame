import torch
import numpy as np
import json
import os
from chess_board import ChessBoard, Position, PieceType, PieceColor
from alpha_zero_lite.board_converter import board_to_input, move_to_policy_idx

# Constants
BOARD_SIZE = 8
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 4096 possible moves

class SelfPlayDataset(torch.utils.data.Dataset):
    """Dataset for training from self-play games"""
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def load_training_data(data_path, device="cpu"):
    """
    Load training data from game records and convert to tensors
    """
    print(f"Loading training data from {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    states = []
    policies = []
    values = []
    
    num_games = len(data['games'])
    print(f"Processing {num_games} games...")
    
    for game_idx, game in enumerate(data['games']):
        # Get the game result for value targets
        result = game['result']  # 1.0 for white win, 0.0 for draw, -1.0 for black win
        
        for move_data in game['moves']:
            # Create a ChessBoard from the board data
            board = recreate_board_from_data(move_data['board'])
            
            # Convert board to input tensor
            state_tensor = board_to_input(board, device).cpu()
            states.append(state_tensor.squeeze(0))
            
            # Policy from visit counts
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            
            if 'visit_counts' in move_data:
                total_visits = 0
                for move_str, count in move_data['visit_counts'].items():
                    from_coords = (int(move_str[0]), int(move_str[1]))
                    to_coords = (int(move_str[2]), int(move_str[3]))
                    
                    # Convert to policy index with the simpler encoding
                    idx = move_to_policy_idx(
                        Position(from_coords[0], from_coords[1]),
                        Position(to_coords[0], to_coords[1])
                    )
                    
                    policy[idx] = count
                    total_visits += count
                
                # Normalize
                if total_visits > 0:
                    policy /= total_visits
            
            policies.append(policy)
            
            # Value target
            player = move_data['player']  # 1 for white, -1 for black
            
            # Adjust value target based on who is to play
            if result == 1.0:  # White won
                value = 1.0 if player == 1 else -1.0
            elif result == -1.0:  # Black won
                value = -1.0 if player == 1 else 1.0
            else:  # Draw
                value = 0.0
            
            values.append(value)
        
        # Progress update
        if (game_idx + 1) % max(1, num_games // 10) == 0:
            print(f"Processed {game_idx + 1}/{num_games} games")
    
    # Convert to tensors
    states_tensor = torch.stack(states)
    policies_tensor = torch.FloatTensor(np.array(policies))
    values_tensor = torch.FloatTensor(np.array(values))
    
    print(f"Created dataset with {len(states)} positions")
    
    return SelfPlayDataset(states_tensor, policies_tensor, values_tensor)


def recreate_board_from_data(board_data):
    """
    Recreate a ChessBoard instance from serialized data
    """
    board = ChessBoard()
    
    # Clear the board first (to avoid interference from initial positions)
    for row in range(8):
        for col in range(8):
            board.set_piece(Position(row, col), ChessBoard.Piece(PieceType.EMPTY, PieceColor.NONE))
    
    # Set pieces according to board data
    board_array = board_data['board']
    for row in range(8):
        for col in range(8):
            piece_data = board_array[row][col]
            piece_type = getattr(PieceType, piece_data['type'].upper()) if piece_data['type'] != 'empty' else PieceType.EMPTY
            piece_color = getattr(PieceColor, piece_data['color'].upper()) if piece_data['color'] != 'none' else PieceColor.NONE
            has_moved = piece_data.get('has_moved', False)
            
            board.set_piece(Position(row, col), ChessBoard.Piece(piece_type, piece_color, has_moved))
    
    # Set other board state
    board.turn = getattr(PieceColor, board_data['turn'].upper())
    board.half_move_clock = board_data.get('half_move_clock', 0)
    
    # Reconstruct last move if available
    if board_data.get('last_move'):
        last_move_data = board_data['last_move']
        start_pos = Position(last_move_data['start_pos'][0], last_move_data['start_pos'][1])
        end_pos = Position(last_move_data['end_pos'][0], last_move_data['end_pos'][1])
        
        piece_type = getattr(PieceType, last_move_data['piece_type'].upper())
        piece_color = getattr(PieceColor, last_move_data['piece_color'].upper())
        piece = ChessBoard.Piece(piece_type, piece_color)
        
        last_move = Move(
            start_pos=start_pos,
            end_pos=end_pos,
            piece=piece,
            is_castling=last_move_data.get('is_castling', False),
            is_en_passant=last_move_data.get('is_en_passant', False)
        )
        
        board.last_move = last_move
    
    return board