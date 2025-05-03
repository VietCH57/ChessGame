from chess_ai_interface import ChessAIInterface
from simple_chess_ai import RandomChessAI, PrioritizeCapturingAI
from src.chess_board import PieceColor

def main():
    # Create the chess AI interface
    interface = ChessAIInterface()
    
    # Create AI players
    interface.register_ai(PieceColor.WHITE, RandomChessAI("Random White"))
    interface.register_ai(PieceColor.BLACK, PrioritizeCapturingAI("Capturing Black"))
    
    # Start the game
    print("Starting chess game...")
    interface.run()

if __name__ == "__main__":
    main()