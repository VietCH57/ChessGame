from sample_ai import RandomAI, SimpleChessAI
from chess_game import ChessGame
from chess_board import PieceColor


if __name__ == "__main__":
    game = ChessGame()
    
    # Human vs AI
    # game.toggle_ai(white_ai=AI())
    game.toggle_ai(black_ai=RandomAI())
    
    # Ai vs AI
    # game.toggle_ai(white_ai=AI1(), black_ai=AI2())
    
    # Run the game
    game.run()