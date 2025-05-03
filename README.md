# ChessGame
How to install: \
1 - Run this in any directory you want: \
git clone https://github.com/VietCH57/ChessGame.git \
2 - cd to ChessGame, then install the pygame library: \
pip install pygame \
3 - Use python to run the game: \
python src/main.py

# UPDATE: Added an interface for AIs

Usage examples are included in this repository (check simple_chess_ai.py and chess_ai_demo.py)

# YOU MUST INCLUDE `get_move` IN YOUR PROGRAM

Available Methods:

`register_ai(self, color, ai_controller):` \
    """
    Register an AI to control a specific color.
    
    Parameters:
    - color: PieceColor.WHITE or PieceColor.BLACK
    - ai_controller: An object with a get_move(board_state, valid_moves) method
    
    Example:
    interface.register_ai(PieceColor.BLACK, MyCustomAI())
    """

`run(self):` \
    """
    Start the main game loop with visualization.
    This method blocks until the game window is closed.
    
    Example:
    interface.run()
    """

`reset_game(self):` \
    """
    Reset the game to the initial state.
    Used to restart the game after completion or to initialize a new game.
    
    Example:
    interface.reset_game()
    """

`get_board_state(self):` \
    """
    Get the current board state as a 2D array of piece information.
    
    Returns:
    A 2D array (8x8) where each cell contains a dictionary with:
    - "type": The type of piece (e.g., "pawn", "rook")
    - "color": The color of the piece (e.g., "white", "black")
    - "has_moved": Boolean indicating if the piece has moved
    
    Example:
    board_state = interface.get_board_state()
    piece_at_a1 = board_state[7][0]  # Bottom-left corner
    """

`get_valid_moves_for_color(self, color):` \
    """
    Get all valid moves for a specific color.
    
    Parameters:
    - color: PieceColor enum value (PieceColor.WHITE or PieceColor.BLACK)
    
    Returns:
    A list of Move objects representing all valid moves for the specified color.
    
    Example:
    valid_moves = interface.get_valid_moves_for_color(PieceColor.WHITE)
    """

`get_move(self, board_state, valid_moves):` \
    """
    Choose the next move for the AI.
    
    Parameters:
    - board_state: 2D array representing the current board state
    - valid_moves: List of valid Move objects the AI can choose from
    
    Returns:
    - A Move object from valid_moves representing the chosen move
    
    This method is called by the interface when it's your AI's turn to move.
    Your AI must return one of the Move objects from the valid_moves list.
    """