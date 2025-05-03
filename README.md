# ChessGame
How to install: \
1 - Chạy lệnh sau ở folder bạn muốn: \
git clone https://github.com/VietCH57/ChessGame.git \
2 - Chuyển directory sang ChessGame, rồi chạy \
pip install pygame \
3 - Chơi game bằng: \
python src/main.py

# UPDATE: Added an interface for AIs
Available Methods

`__init__()`
Initialize a new chess game interface with a fresh board.

`reset_game()`
Reset the game to its initial state.
Returns the new game state.

`get_board_state()`
Get the current board state as a 2D array of piece dictionaries.
Each piece has `type`, `color`, and `has_moved` properties.

`get_piece_at_position(row, col)`
Get information about a piece at a specific position.
Returns a dictionary with `type`, `color`, and `has_moved` properties.

`get_valid_moves(row, col)`
Get all valid moves for a piece at the specified position.
Returns a list of move dictionaries.

`make_move(start_row, start_col, end_row, end_col)`
Make a move on the board.
Returns a dictionary with `success`, `move_executed`, and `game_state` keys.

`get_current_turn()`
Get the current player's turn ("white" or "black").

`is_game_over()`
Check if the game is over due to checkmate, stalemate, or draws.
Returns a dictionary with `game_over` and (if game is over) `result` keys.

`get_fen()`
Get the FEN notation of the current board state.