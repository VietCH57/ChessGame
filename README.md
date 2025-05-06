### ChessGame
Tải xuống: \
1 - Clone repo này: \
git clone https://github.com/VietCH57/ChessGame.git \
2 - Chuyển đến thư mục ChessGame rồi tải thư viện pygame: \
pip install pygame \
3 - Chơi game: \
python src/main.py

### UPDATE: Thêm interface cho AI

Cách viết AI:
# 1. Import các module cần thiết
```python
from chess_board import ChessBoard, Position, PieceType, PieceColor
from chess_ai_interface import ChessAI
```

# 2. Tạo class kế thừa ChessAI
```python
class MyChessAI(ChessAI):
    def __init__(self):
        # Khởi tạo AI của bạn
        pass
        
    def get_move(self, board: ChessBoard, color: PieceColor):
        # Logic của AI để chọn nước đi
        # Trả về: tuple(Position từ, Position đến)
        pass
```

# 3. Chạy AI của bạn
```python
from chess_game_with_ai import ChessGame

game = ChessGame()
game.toggle_ai(white_ai=MyChessAI())  # Cho AI chơi quân trắng
game.run()
```

### Thành phần cơ bản của bàn cờ:
Vị trí:
- position = Position(row, col)  # row và col từ 0-7

Các loại quân cờ:
- PieceType.PAWN    # Tốt
- PieceType.ROOK    # Xe
- PieceType.KNIGHT  # Mã
- PieceType.BISHOP  # Tượng
- PieceType.QUEEN   # Hậu
- PieceType.KING    # Vua
- PieceType.EMPTY   # Ô trống

Màu quân cờ:
- PieceColor.WHITE  # Trắng
- PieceColor.BLACK  # Đen
- PieceColor.NONE   # Không màu (cho ô trống)

### Đối tượng Move
Mỗi nước đi trong danh sách valid_moves là một đối tượng Move với các thuộc tính:
- start_pos: Vị trí xuất phát
- end_pos: Vị trí đích
- piece: Quân cờ được di chuyển
- captured_piece: Quân bị bắt (nếu có)
- is_castling: True nếu là nước nhập thành
- is_en_passant: True nếu là nước bắt tốt qua đường

### Các methods hữu ích từ class ChessBoard:
- board.get_piece(position): Lấy quân cờ tại vị trí
- board.get_valid_moves(position): Lấy danh sách các nước đi hợp lệ từ vị trí
- board.is_check(color): Kiểm tra xem vua có đang bị chiếu không
- board.copy_board(): Tạo bản sao của bàn cờ (để thử nghiệm nước đi)

### Các files mẫu:
Tôi có viết một số files mẫu để mọi người có thể tham khảo cấu trúc: 
- ai_template.py 
- run.py 
