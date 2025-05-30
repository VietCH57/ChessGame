import json
import traceback
from chess_board import ChessBoard, Position, PieceColor, Move

class HeadlessChessGame:
    """A non-graphical chess game for AI training"""
    
    def __init__(self):
        self.board = None
        self.game_over = False
        self.result = None
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = ChessBoard()
        self.game_over = False
        self.result = None
    
    def check_game_state(self):
        """Check if the game is over"""
        try:
            # Check for checkmate
            if self.board.is_checkmate(PieceColor.WHITE):
                self.game_over = True
                self.result = "black_wins"
                return True
            elif self.board.is_checkmate(PieceColor.BLACK):
                self.game_over = True
                self.result = "white_wins"
                return True
            
            # Check for stalemate
            if self.board.is_stalemate(PieceColor.WHITE) or self.board.is_stalemate(PieceColor.BLACK):
                self.game_over = True
                self.result = "draw"
                return True
            
            # Check for fifty-move rule
            if self.board.is_fifty_move_rule_draw():
                self.game_over = True
                self.result = "draw"
                return True
            
            # Check for threefold repetition
            if self.board.is_threefold_repetition():
                self.game_over = True
                self.result = "draw"
                return True
            
            return False
        except Exception as e:
            print(f"Error checking game state: {e}")
            # If we can't check game state, just end the game as a draw
            self.game_over = True
            self.result = "draw"
            return True
    
    def run_single_game(self, white_ai, black_ai, max_moves=200, collect_data=False):
        """Run a single game between two AIs"""
        self.reset()
        moves = 0
        game_data = {
            "moves": [],
            "result": None
        }
        
        while not self.game_over and moves < max_moves:
            # Get current state
            current_color = self.board.turn
            
            # Get AI for current player
            ai = white_ai if current_color == PieceColor.WHITE else black_ai
            
            # Record state before move if collecting data
            if collect_data:
                state_data = self.get_board_data()
            
            try:
                # Get and apply move
                from_pos, to_pos = ai.get_move(self.board, current_color)
                move_result = self.board.move_piece(from_pos, to_pos)
                
                if move_result is None:
                    print(f"Invalid move attempted: {from_pos.row},{from_pos.col} to {to_pos.row},{to_pos.col}")
                    # End game as a draw if an invalid move is attempted
                    self.game_over = True
                    self.result = "draw"
                    break
                
                # Record move if collecting data
                if collect_data:
                    move_data = {
                        "from": [from_pos.row, from_pos.col],
                        "to": [to_pos.row, to_pos.col],
                        "player": 1 if current_color == PieceColor.WHITE else -1,
                        "board": state_data
                    }
                    
                    # Try to get MCTS visit counts
                    try:
                        if hasattr(ai, 'mcts') and hasattr(ai.mcts, 'tree'):
                            visit_counts = {}
                            # Look for node data
                            for node in ai.mcts.tree.values():
                                if 'children_data' in node:
                                    for child in node['children_data']:
                                        if 'move' in child and 'visits' in child:
                                            from_pos, to_pos = child['move']
                                            move_str = f"{from_pos.row}{from_pos.col}{to_pos.row}{to_pos.col}"
                                            visit_counts[move_str] = child['visits']
                            
                            if visit_counts:
                                move_data["visit_counts"] = visit_counts
                    except Exception as e:
                        print(f"Error recording visit counts: {e}")
                    
                    game_data["moves"].append(move_data)
                
                # Check for game over
                self.check_game_state()
                
                # Count the move
                moves += 1
                
            except Exception as e:
                print(f"Error during move: {e}")
                traceback.print_exc()
                # End game as a draw if there's an error
                self.game_over = True
                self.result = "draw"
                break
        
        # Handle game over by move limit
        if moves >= max_moves and not self.game_over:
            self.game_over = True
            self.result = "draw"
        
        # Record result
        if collect_data:
            if self.result == "white_wins":
                game_data["result"] = 1.0
            elif self.result == "black_wins":
                game_data["result"] = -1.0
            else:  # draw
                game_data["result"] = 0.0
        
        return {
            "result": self.result,
            "moves": moves,
            "game_data": game_data if collect_data else None
        }
    
    def run_many_games(self, white_ai, black_ai, num_games=100, max_moves=200, 
                      collect_data=False, swap_sides=True, output_file=None):
        """Run multiple games between two AIs"""
        stats = {
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "games": []
        }
        
        for i in range(num_games):
            try:
                # Swap sides if enabled (every other game)
                if swap_sides and i % 2 == 1:
                    current_white = black_ai
                    current_black = white_ai
                    sides_swapped = True
                else:
                    current_white = white_ai
                    current_black = black_ai
                    sides_swapped = False
                
                # Run game
                result = self.run_single_game(
                    current_white, current_black, max_moves, collect_data
                )
                
                # Update stats based on result
                game_result = result["result"]
                
                if game_result == "white_wins":
                    stats["white_wins"] += 1
                    if sides_swapped and collect_data:
                        # Original black AI won while playing as white
                        result["game_data"]["result"] = -1.0 if not sides_swapped else 1.0
                elif game_result == "black_wins":
                    stats["black_wins"] += 1
                    if sides_swapped and collect_data:
                        # Original white AI won while playing as black
                        result["game_data"]["result"] = 1.0 if not sides_swapped else -1.0
                else:  # draw
                    stats["draws"] += 1
                
                # Store game data
                if collect_data and result["game_data"]:
                    stats["games"].append(result["game_data"])
                
                # Log progress
                print(f"Game {i+1}/{num_games}: {game_result} in {result['moves']} moves" + 
                      (" (sides swapped)" if sides_swapped else ""))
                      
            except Exception as e:
                print(f"Error running game {i+1}: {e}")
                traceback.print_exc()
                stats["draws"] += 1
        
        # Calculate win rates
        total_games = num_games
        stats["white_win_rate"] = stats["white_wins"] / total_games
        stats["black_win_rate"] = stats["black_wins"] / total_games
        stats["draw_rate"] = stats["draws"] / total_games
        
        # Save data if output file is provided
        if output_file and collect_data:
            try:
                with open(output_file, 'w') as f:
                    json.dump(stats, f)
                print(f"Game data saved to {output_file}")
            except Exception as e:
                print(f"Error saving game data: {e}")
        
        return stats
    
    def get_board_data(self):
        """Get a serializable representation of the board state"""
        data = []
        for row in range(8):
            row_data = []
            for col in range(8):
                piece = self.board.get_piece(Position(row, col))
                if piece:
                    row_data.append({
                        "type": piece.type.value,
                        "color": piece.color.value,
                        "has_moved": piece.has_moved
                    })
                else:
                    row_data.append({
                        "type": "empty",
                        "color": "none",
                        "has_moved": False
                    })
            data.append(row_data)
        
        return {
            "board": data,
            "turn": self.board.turn.value,
            "last_move": self._serialize_move(self.board.last_move) if self.board.last_move else None,
            "half_move_clock": self.board.half_move_clock
        }
    
    def _serialize_move(self, move):
        """Convert a Move object to a serializable dict"""
        if not move:
            return None
        
        try:
            return {
                "start_pos": [move.start_pos.row, move.start_pos.col],
                "end_pos": [move.end_pos.row, move.end_pos.col],
                "piece_type": move.piece.type.value,
                "piece_color": move.piece.color.value,
                "is_castling": move.is_castling,
                "is_en_passant": move.is_en_passant
            }
        except Exception as e:
            print(f"Error serializing move: {e}")
            return None