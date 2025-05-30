import torch
import numpy as np
import os
import random
from interface import ChessAI
from chess_board import ChessBoard, Position, PieceColor, PieceType, Move
from alpha_zero_lite.network import AlphaZeroLiteNetwork
from alpha_zero_lite.mcts import MCTS
from alpha_zero_lite.board_converter import board_to_input

class AlphaZeroLite(ChessAI):
    """
    A lightweight AlphaZero implementation that implements the ChessAI interface
    """
    def __init__(self, model_path=None, exploration_factor=1.0, num_simulations=100, 
                 temperature=1.0, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize network
        self.network = AlphaZeroLiteNetwork(blocks=5, filters=64).to(self.device)
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # MCTS parameters
        self.exploration_factor = exploration_factor
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        # Initialize MCTS
        self.mcts = MCTS(
            self.network,
            num_simulations=self.num_simulations,
            exploration_factor=self.exploration_factor,
            device=self.device
        )
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> tuple[Position, Position]:
        """
        Get the next move from the AI using MCTS with neural network guidance.
        Implements the ChessAI interface method.
        """
        # First get all valid moves for this color
        valid_moves = self._get_all_valid_moves(board, color)
        
        if not valid_moves:
            print("No valid moves available!")
            return Position(0, 0), Position(0, 0)
            
        try:
            # Try to run MCTS to get visit counts for each move
            root_node = self.mcts.search(board)
            
            if root_node is None or not root_node.get('children_data', []):
                # If MCTS fails or returns empty data, use random valid move
                print("MCTS search failed or returned no moves, using random valid move")
                return random.choice(valid_moves)
            
            # For move selection with temperature
            visit_counts = np.array([child['visits'] for child in root_node['children_data']])
            moves = [child['move'] for child in root_node['children_data']]
            
            # Check if arrays are empty
            if len(visit_counts) == 0 or len(moves) == 0:
                print("No visit counts or moves available, using random valid move")
                return random.choice(valid_moves)
            
            if self.temperature == 0:
                # Deterministic selection (pick most visited)
                best_idx = np.argmax(visit_counts)
                selected_move = moves[best_idx]
            else:
                # Sample proportionally to adjusted visit counts
                adjusted_counts = np.power(visit_counts, 1/self.temperature)
                # Avoid division by zero
                sum_counts = np.sum(adjusted_counts)
                if sum_counts > 0:
                    adjusted_counts = adjusted_counts / sum_counts  # Normalize to probabilities
                else:
                    # Equal probability if all counts are zero
                    adjusted_counts = np.ones_like(adjusted_counts) / len(adjusted_counts)
                
                selected_idx = np.random.choice(len(moves), p=adjusted_counts)
                selected_move = moves[selected_idx]
            
            return selected_move[0], selected_move[1]  # from_pos, to_pos
            
        except Exception as e:
            # If anything goes wrong, fall back to random move
            print(f"Error in get_move: {e}")
            return random.choice(valid_moves)
    
    def save_model(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _get_all_valid_moves(self, board, color):
        """Get all valid moves for a color"""
        moves = []
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece and piece.color == color:
                    valid_moves = board.get_valid_moves(pos)
                    for move in valid_moves:
                        moves.append((move.start_pos, move.end_pos))
        return moves