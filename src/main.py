import pygame
import sys
import threading
from chess_game import ChessGame
import chess_api  # Import the new API module

def start_api_server():
    """Start the API server in a separate thread"""
    chess_api.app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    # Start API server in a separate thread
    api_thread = threading.Thread(target=start_api_server)
    api_thread.daemon = True  # Thread will exit when the main program exits
    api_thread.start()
    
    # Start the game UI as usual
    game = ChessGame()
    game.run()