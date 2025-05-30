import os
import argparse
import torch
import time
from alpha_zero_lite import AlphaZeroLite, AlphaZeroLiteTrainer, HeadlessChessGame
from alpha_zero_lite.dataset import load_training_data
from interface import ChessAI

def self_play(args):
    """Run self-play games to generate training data"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize AI
    ai = AlphaZeroLite(
        model_path=args.model_path,
        num_simulations=args.simulations,
        exploration_factor=args.exploration,
        temperature=args.temperature
    )
    
    # Create headless game engine
    game_engine = HeadlessChessGame()
    
    # Run self-play games
    print(f"Running {args.num_games} self-play games...")
    stats = game_engine.run_many_games(
        white_ai=ai,
        black_ai=ai,
        num_games=args.num_games,
        max_moves=args.max_moves,
        collect_data=True,
        swap_sides=True,
        output_file=os.path.join(args.output_dir, "self_play_data.json")
    )
    
    # Print results
    print("Self-play complete!")
    print(f"White wins: {stats['white_wins']} ({stats['white_win_rate']:.2%})")
    print(f"Black wins: {stats['black_wins']} ({stats['black_win_rate']:.2%})")
    print(f"Draws: {stats['draws']} ({stats['draw_rate']:.2%})")
    print(f"Data saved to {os.path.join(args.output_dir, 'self_play_data.json')}")


def train(args):
    """Train the model using generated data"""
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize AI with existing model if provided
    model = AlphaZeroLite(
        model_path=args.model_path,
        device=args.device
    )
    
    # Load training data
    dataset = load_training_data(
        args.data_path,
        device="cpu"  # Load on CPU first, then transfer in batches
    )
    
    # Initialize trainer
    trainer = AlphaZeroLiteTrainer(
        model=model,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        lr=args.learning_rate
    )
    
    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()
    losses = trainer.train_epoch(
        dataset=dataset,
        num_epochs=args.epochs,
        checkpoint_dir=args.model_dir,
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, "alpha_zero_lite_final.pt")
    model.save_model(final_model_path)
    
    print(f"Training complete in {time.time() - start_time:.2f} seconds!")
    print(f"Final model saved to {final_model_path}")


def evaluate(args):
    """Evaluate the model against a baseline"""
    # Create output directory for results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the trained model
    trained_model = AlphaZeroLite(
        model_path=args.model_path,
        num_simulations=args.simulations,
        temperature=0.0  # Use deterministic selection for evaluation
    )
    
    # Initialize the baseline model or a random player
    if args.baseline_path:
        baseline_model = AlphaZeroLite(
            model_path=args.baseline_path,
            num_simulations=args.simulations,
            temperature=0.0
        )
    else:
        # Use a simple random AI
        baseline_model = RandomChessAI()
    
    # Create headless game engine
    game_engine = HeadlessChessGame()
    
    # Run evaluation games
    print(f"Evaluating against {'baseline model' if args.baseline_path else 'random AI'}...")
    stats = game_engine.run_many_games(
        white_ai=trained_model,
        black_ai=baseline_model,
        num_games=args.num_games,
        max_moves=args.max_moves,
        collect_data=False,
        swap_sides=True,
        output_file=os.path.join(args.output_dir, "evaluation_results.json")
    )
    
    # Print results
    print("Evaluation complete!")
    print(f"Trained model wins: {stats['white_wins'] + stats['black_wins'] // 2} " +
          f"({(stats['white_wins'] + stats['black_wins'] // 2) / args.num_games:.2%})")
    print(f"Baseline wins: {(stats['black_wins'] + stats['white_wins'] // 2)} " +
          f"({(stats['black_wins'] + stats['white_wins'] // 2) / args.num_games:.2%})")
    print(f"Draws: {stats['draws']} ({stats['draw_rate']:.2%})")


# Simple random baseline AI
class RandomChessAI(ChessAI):
    """A simple AI that makes random moves"""
    def __init__(self):
        import random
        self.random = random
    
    def get_move(self, board, color):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece.color == color:
                    moves = board.get_valid_moves(pos)
                    valid_moves.extend([(move.start_pos, move.end_pos) for move in moves])
        
        if valid_moves:
            return self.random.choice(valid_moves)
        
        # No valid moves, return dummy move
        return Position(0, 0), Position(0, 0)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate AlphaZeroLite")
    
    # Common arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to existing model to load")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Self-play arguments
    sp_parser = subparsers.add_parser("self-play", help="Generate self-play data")
    sp_parser.add_argument("--output-dir", type=str, default="training_data",
                          help="Directory to save self-play data")
    sp_parser.add_argument("--num-games", type=int, default=100,
                          help="Number of self-play games to run")
    sp_parser.add_argument("--max-moves", type=int, default=200,
                          help="Maximum number of moves per game")
    sp_parser.add_argument("--simulations", type=int, default=100,
                          help="Number of MCTS simulations per move")
    sp_parser.add_argument("--exploration", type=float, default=1.0,
                          help="Exploration factor for MCTS")
    sp_parser.add_argument("--temperature", type=float, default=1.0,
                          help="Temperature for move selection")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train model on generated data")
    train_parser.add_argument("--data-path", type=str, required=True,
                             help="Path to self-play data JSON file")
    train_parser.add_argument("--model-dir", type=str, default="models",
                             help="Directory to save trained models")
    train_parser.add_argument("--epochs", type=int, default=10,
                             help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=256,
                             help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, default=0.001,
                             help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4,
                             help="Weight decay for Adam optimizer")
    train_parser.add_argument("--checkpoint-freq", type=int, default=1,
                             help="Save model every N epochs")
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--baseline-path", type=str, default=None,
                            help="Path to baseline model (uses random AI if not provided)")
    eval_parser.add_argument("--output-dir", type=str, default="evaluation",
                            help="Directory to save evaluation results")
    eval_parser.add_argument("--num-games", type=int, default=20,
                            help="Number of evaluation games")
    eval_parser.add_argument("--max-moves", type=int, default=200,
                            help="Maximum number of moves per game")
    eval_parser.add_argument("--simulations", type=int, default=100,
                            help="Number of MCTS simulations per move")
    
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == "self-play":
        self_play(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        parser.print_help()