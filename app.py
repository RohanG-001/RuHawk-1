from flask import Flask, render_template, request, jsonify
import chess
import torch
import torch.nn as nn

app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ChessBot class
class ChessBot(nn.Module):
    def __init__(self):
        super(ChessBot, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.symbol()
            channel = piece_to_index[piece_type]
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[channel, row, col] = 1
    return tensor

def get_legal_moves(board):
    return list(board.legal_moves)

def make_move(board, move):
    board = board.copy()
    board.push(move)
    return board

def evaluate_position(board, model):
    if board.is_checkmate():
        return -1000 if board.turn == chess.WHITE else 1000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    tensor = board_to_tensor(board)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        score = model(tensor)
    return score.item()

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model), None
    
    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in get_legal_moves(board):
            new_board = make_move(board, move)
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False, model)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in get_legal_moves(board):
            new_board = make_move(board, move)
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True, model)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

# Load model
model = ChessBot()
model.load_state_dict(torch.load('best_chess_model.pth', map_location=device))
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/move', methods=['POST'])
def get_bot_move():
    try:
        data = request.json
        fen = data.get('fen', chess.Board().fen())
        board = chess.Board(fen)
        
        print(f"Received FEN: {fen}")
        print(f"Bot is thinking...")
        
        _, move = minimax(board, 3, float('-inf'), float('inf'), False, model)
        
        print(f"Bot move: {move}")
        print(f"Move in SAN: {board.san(move)}")
        
        return jsonify({
            'move': board.san(move),
            'from': chess.square_name(move.from_square),
            'to': chess.square_name(move.to_square)
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/new_game', methods=['POST'])
def new_game():
    board = chess.Board()
    return jsonify({'fen': board.fen()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
