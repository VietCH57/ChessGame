from chess_game_with_ai import ChessGame
from chess_ai_interface import RandomChessAI
from ai_template import MyChessAI
from chess_board import PieceColor

def main():
    """
    Script chạy game cờ vua với AI tùy chỉnh.
    """
    # Khởi tạo game
    game = ChessGame()
    
    # Tạo AI của bạn
    my_ai = MyChessAI()
    
    # Cấu hình game để sử dụng AI của bạn (quân trắng)
    game.toggle_ai(white_ai=my_ai)
    
    # HOẶC, để AI của bạn chơi với AI kháckhác
    # game.toggle_ai(white_ai=my_ai, black_ai=other_aiai())
    
    # HOẶC, để chơi lại chính AI của bạn
    # game.toggle_ai(black_ai=my_ai)
    
    """
    Tóm lại là game.toggle_ai cho phép bạn chọn phe nào là AI, phe nào được bỏ trống thì sẽ là người chơichơi
    """
    # Chạy game
    game.run()

if __name__ == "__main__":
    main()