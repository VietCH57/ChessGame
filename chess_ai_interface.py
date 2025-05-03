import sys
import os
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import game components
from src.chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move
import pygame

class ChessAIInterface:
    """
    A direct interface for AIs to interact with the chess game.
    """
    def __init__(self):
        """Initialize a new chess AI interface"""
        pygame.init()
        self.screen = pygame.display.set_mode((8*80, 8*80))  # Using BOARD_SIZE and SQUARE_SIZE values
        pygame.display.set_caption("Chess Game with AI")
        self.clock = pygame.time.Clock()
        
        # Create a new chess board
        self.board = ChessBoard()
        
        # AI players
        self.white_ai = None
        self.black_ai = None
        
        # Game state
        self.game_over = False
        self.result_message = ""
        self.selected_position = None
        
        # Load chess piece images
        self.images = self._load_piece_images()
        
    def register_ai(self, color, ai_controller):
        """
        Register an AI to control a specific color
        
        Parameters:
        - color: PieceColor.WHITE or PieceColor.BLACK
        - ai_controller: An object with a get_move(board_state, valid_moves) method
        """
        if color == PieceColor.WHITE:
            self.white_ai = ai_controller
        elif color == PieceColor.BLACK:
            self.black_ai = ai_controller
        else:
            raise ValueError("Color must be WHITE or BLACK")
    
    def _load_piece_images(self):
        """Load or create placeholder images for chess pieces"""
        images = {}
        piece_types = [p.value for p in PieceType if p != PieceType.EMPTY]
        colors = [c.value for c in PieceColor if c != PieceColor.NONE]
        
        for piece_type in piece_types:
            for color in colors:
                image_path = f"assets/{color}_{piece_type}.png"
                try:
                    images[f"{color}_{piece_type}"] = pygame.transform.scale(
                        pygame.image.load(image_path),
                        (80, 80)  # SQUARE_SIZE
                    )
                except (pygame.error, FileNotFoundError):
                    # Create a placeholder image with text
                    img = pygame.Surface((80, 80), pygame.SRCALPHA)
                    font = pygame.font.SysFont("Arial", 20)
                    text = font.render(f"{color[0]}{piece_type[0]}", True, (0, 0, 0))
                    img.blit(text, text.get_rect(center=(80/2, 80/2)))
                    images[f"{color}_{piece_type}"] = img
        return images
    
    def draw_board(self):
        """Draw the chess board with pieces"""
        # Draw the board squares
        for row in range(8):
            for col in range(8):
                # Determine square color (light or dark)
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                
                # Highlight selected piece
                if self.selected_position and row == self.selected_position.row and col == self.selected_position.col:
                    color = (124, 192, 214)  # Highlight selected
                
                # Draw square
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    pygame.Rect(col * 80, row * 80, 80, 80)
                )
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row][col]
                if piece.type != PieceType.EMPTY:
                    piece_img = self.images.get(f"{piece.color.value}_{piece.type.value}")
                    if piece_img:
                        self.screen.blit(
                            piece_img, 
                            pygame.Rect(col * 80, row * 80, 80, 80)
                        )
        
        # Draw game over message if applicable
        if self.game_over:
            self._draw_game_over_message()
    
    def _draw_game_over_message(self):
        """Draw the game over message"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((8*80, 8*80), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Black with alpha
        self.screen.blit(overlay, (0, 0))
        
        # Create message box
        font = pygame.font.SysFont('Arial', 36)
        text_surface = font.render(self.result_message, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(8*80 // 2, 8*80 // 2))
        
        # Create border around text
        border_rect = text_rect.copy()
        border_rect.inflate_ip(40, 40)
        pygame.draw.rect(self.screen, (255, 255, 255), border_rect, border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), border_rect.inflate(-6, -6), border_radius=8)
        
        # Draw text
        self.screen.blit(text_surface, text_rect)
        
        # Add instructions to restart
        restart_font = pygame.font.SysFont('Arial', 24)
        restart_text = restart_font.render("Press R to restart game", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(8*80 // 2, 8*80 // 2 + 50))
        self.screen.blit(restart_text, restart_rect)
    
    def handle_click(self, pos):
        """Handle mouse clicks on the board"""
        if self.game_over:
            return
            
        col = pos[0] // 80
        row = pos[1] // 80
        
        clicked_position = Position(row, col)
        clicked_piece = self.board.get_piece(clicked_position)
        
        if self.selected_position:
            # If a piece is already selected, try to move it
            if clicked_position.row == self.selected_position.row and clicked_position.col == self.selected_position.col:
                # Deselect if clicking the same position
                self.selected_position = None
            else:
                # Try to move selected piece to the clicked position
                move = self.board.move_piece(self.selected_position, clicked_position)
                if move:
                    # Check game state after move
                    self._check_game_state()
                self.selected_position = None
        else:
            # Select a piece if it belongs to the current player
            if clicked_piece.type != PieceType.EMPTY and clicked_piece.color == self.board.turn:
                self.selected_position = clicked_position
    
    def _check_game_state(self):
        """Check for checkmate, stalemate, and draws after a move"""
        opponent_color = PieceColor.BLACK if self.board.turn == PieceColor.WHITE else PieceColor.WHITE
        
        if self.board.is_checkmate(self.board.turn):
            self.game_over = True
            self.result_message = f"Checkmate! {opponent_color.value.capitalize()} wins!"
        elif self.board.is_stalemate(self.board.turn):
            self.game_over = True
            self.result_message = "Stalemate! Draw."
        elif self.board.is_fifty_move_rule_draw():
            self.game_over = True
            self.result_message = "Draw by fifty-move rule."
        elif self.board.is_threefold_repetition():
            self.game_over = True
            self.result_message = "Draw by threefold repetition."
    
    def reset_game(self):
        """Reset the game to the initial state"""
        self.board = ChessBoard()
        self.selected_position = None
        self.game_over = False
        self.result_message = ""
    
    def get_board_state(self):
        """Get the current board state as a 2D array"""
        board_state = []
        for row in range(8):
            row_data = []
            for col in range(8):
                piece = self.board.board[row][col]
                row_data.append({
                    "type": piece.type.value,
                    "color": piece.color.value,
                    "has_moved": piece.has_moved
                })
            board_state.append(row_data)
        return board_state
    
    def get_valid_moves_for_color(self, color):
        """Get all valid moves for a specific color"""
        all_moves = []
        for row in range(8):
            for col in range(8):
                position = Position(row, col)
                piece = self.board.get_piece(position)
                if piece.color == color:
                    moves = self.board.get_valid_moves(position)
                    all_moves.extend(moves)
        return all_moves
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            # Process AI moves if it's an AI's turn
            current_turn = self.board.turn
            ai_player = None
            
            if current_turn == PieceColor.WHITE and self.white_ai:
                ai_player = self.white_ai
            elif current_turn == PieceColor.BLACK and self.black_ai:
                ai_player = self.black_ai
            
            # If it's an AI's turn, get and execute its move
            if not self.game_over and ai_player:
                # Get board state for the AI
                board_state = self.get_board_state()
                
                # Get valid moves for the current color
                valid_moves = self.get_valid_moves_for_color(current_turn)
                
                if valid_moves:
                    # Ask the AI for a move
                    ai_move = ai_player.get_move(board_state, valid_moves)
                    
                    if ai_move:
                        start_pos = ai_move.start_pos
                        end_pos = ai_move.end_pos
                        
                        # Execute the move
                        move_result = self.board.move_piece(start_pos, end_pos)
                        if move_result:
                            # Check game state after move
                            self._check_game_state()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Only handle clicks if it's a human player's turn
                        current_turn = self.board.turn
                        is_ai_turn = (current_turn == PieceColor.WHITE and self.white_ai) or \
                                    (current_turn == PieceColor.BLACK and self.black_ai)
                        if not is_ai_turn:
                            self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:  # Restart game with R key
                        self.reset_game()
            
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            # Draw the board with all visual indicators
            self.draw_board()
            
            pygame.display.flip()
            self.clock.tick(60)
            
            # Add a small delay to make AI moves visible
            if not self.game_over and ai_player:
                time.sleep(0.5)
        
        pygame.quit()