from interface import ChessAI
from headless import HeadlessChessGame
from train_ai import ChessTrainer, TrainableChessAI
from example_ai import MinimaxChessAI, NeuralNetworkAI
import time
import random

def main():
    print("CHESS AI TRAINING FRAMEWORK")
    print("==========================\n")
    
    # Khởi tạo trainer
    trainer = ChessTrainer(output_dir="training_data")
    
    # 1. Đánh giá tốc độ chạy của headless mode
    print("Đang kiểm tra tốc độ chạy headless mode...")
    engine = HeadlessChessGame()
    
    # Tạo AI đơn giản
    random_ai = TrainableChessAI(exploration_rate=1.0)  # Luôn chọn ngẫu nhiên
    minimax_ai = MinimaxChessAI(depth=2)
    
    # Đo tốc độ với AI ngẫu nhiên
    start_time = time.time()
    result = engine.run_game(random_ai, random_ai, max_moves=100)
    elapsed = time.time() - start_time
    print(f"100 nước đi ngẫu nhiên: {elapsed:.3f} giây ({result['moves_per_second']:.1f} nước/giây)")
    
    # Đo tốc độ với minimax
    start_time = time.time()
    result = engine.run_game(minimax_ai, random_ai, max_moves=30)
    elapsed = time.time() - start_time
    print(f"30 nước đi minimax (độ sâu 2): {elapsed:.3f} giây ({result['moves_per_second']:.1f} nước/giây)")
    
    # 2. Chạy nhiều game giữa hai AI
    print("\nChạy đánh giá hiệu suất giữa hai AI...")
    stats = trainer.evaluate_ai(minimax_ai, random_ai, num_games=10, max_moves=50)
    
    # 3. Ví dụ về huấn luyện Neural Network AI
    print("\nDemo huấn luyện Neural Network AI...")
    neural_ai = NeuralNetworkAI()
    
    # Giả lập huấn luyện - trong thực tế sẽ cập nhật trọng số model
    def training_callback(ai, game_data, game_index):
        if (game_index + 1) % 5 == 0:
            print(f"  Đã huấn luyện {game_index + 1} trận, đang cập nhật model...")
            # Trong thực tế: cập nhật trọng số của model dựa trên dữ liệu trận đấu
            
    # Huấn luyện với self-play
    print("Huấn luyện AI với self-play (demo 10 trận)...")
    training_stats = trainer.train_self_play(
        neural_ai, 
        num_games=10,
        save_interval=5,
        callback=training_callback
    )
    
    # Lưu model sau khi huấn luyện
    neural_ai.save_model("training_data/final_model")
    
    print("\nĐã hoàn thành demo huấn luyện!")

if __name__ == "__main__":
    main()