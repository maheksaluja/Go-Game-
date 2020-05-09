from copy import deepcopy
import json


class AlphaBeta:
    def __init__(self, board, previous_board, move, lastDiedPieces, pieceType):
        self.board = deepcopy(board)
        self.previous_board = deepcopy(previous_board)
        self.diedPiece = []
        self.lastMove = move
        self.enemyPiecesDied = 0
        self.myPiecesDied = 0
        self.lastDiedPieces = lastDiedPieces
        self.pieceType = pieceType

    def evaluationFunction(self, updatedBoard):
        myPoints = 0
        enemyPoints = 0
        group = []
        for i in range(5):
            for j in range(5):
                if updatedBoard[i][j] == piece_type:
                    myPoints += 1
                if updatedBoard[i][j] == 3 - piece_type:
                    enemyPoints += 1
        myPoints += self.enemyPiecesDied * 16
        enemyPoints += self.myPiecesDied * 10
        if piece_type == 1:
            enemyPoints += 2.5
        else:
            myPoints += 2.5
        return myPoints - enemyPoints

    def avoid_edge_move(self, x, y, dist):
        if x == 0 or x == 4 or y == 0 or y == 4:
            return dist * 2
        return dist

    def getNeighbour(self, pieceType, x, y, board):
        neighbours = self.get_all_neighbours(x, y, board)
        enemycount= 0
        myPieces = 0
        for i, j in neighbours:
            if board[i][j] == 3 - pieceType:
                enemycount += 1
            else:
                myPieces += 1
        return enemycount, myPieces

    def getPossibleMove(self, nBoard, pieceType, player):
        possibleMove = []
        for i in range(5):
            for j in range(5):
                if nBoard[i][j] == 0 and self.is_move_valid(i, j, pieceType, nBoard, player):
                    dist = abs(i - self.lastMove[0]) + abs(j - self.lastMove[1])
                    dist = self.avoid_edge_move(i, j, dist)
                    enemySurrounding, myPieces = self.getNeighbour(pieceType, i, j, nBoard)
                    possibleMove.append((i, j, enemySurrounding, myPieces, dist))
        return sorted(possibleMove, key=lambda x: x[-1])

    def getNearestOpponentPiece(self, x, y, pieceType, nboard):
        d = 10000
        for i in range(5):
            for j in range(5):
                if nboard[i][j] == pieceType:
                    dist = abs(i - x) + abs(j - y)
                    d = min(d, dist)
        return d

    def is_move_inside_board(self, i, j, board):
        if not (0 <= i < len(board)):
            return False
        if not (0 <= j < len(board)):
            return False
        return True

    def is_move_valid(self, i, j, pieceType, nBoard, player):
        if not self.is_move_inside_board(i, j, board):
            return False

        if nBoard[i][j] != 0:
            return False

        test_board = deepcopy(nBoard)

        test_board[i][j] = pieceType
        if self.is_there_liberty(i, j, test_board):
            return True

        diedPiece, test_board = self.removediedpieces(3 - pieceType, test_board)
        if not self.is_there_liberty(i, j, test_board):
            return False

        else:
            if self.is_same_previous_state(test_board, player):
                return False
        return True

    def is_same_previous_state(self, board, player):
        if player == 'max':
            previous_board = self.previous_board
        else:
            previous_board = self.board
        for i in range(5):
            for j in range(5):
                if previous_board[i][j] != board[i][j]:
                    return False
        return True

    def isGameEnd(self, n):
        if n >= 25:
            return True
        return False

    def amIWinner(self, board):
        myScore = 0
        enemyScore = 0
        if self.pieceType == 1:
            enemyScore += 2.5
        else:
            myScore += 2.5
        for i in range(5):
            for j in range(5):
                if board[i][j] == self.pieceType:
                    myScore += 1
                if board[i][j] == 3 - self.pieceType:
                    enemyScore += 1
        if myScore > enemyScore:
            return True
        return False

    def isThisBestPost(self, i, j):
        if (0 < i < 4) and (0 < j < 4):
            return False
        return True

    def min_value(self, alpha, beta, depth, pieceType, board, n):
        enemyPieces = len(self.find_died_pieces(pieceType, board))
        self.enemyPiecesDied += enemyPieces
        diedPiece, newBoard = self.removediedpieces(pieceType, deepcopy(board))
        self.diedPiece = diedPiece
        if self.isGameEnd(n):
            self.enemyPiecesDied -= enemyPieces
            if self.amIWinner(newBoard):
                return 40, None, None
            else:
                return -20, None, None

        if depth == 0:
            value = self.evaluationFunction(newBoard), 0, 0
            self.enemyPiecesDied -= enemyPieces
            return value

        x_pos = None
        y_pos = None
        bestValue = 100000000
        moves = self.getPossibleMove(newBoard, pieceType, 'min')
        counter = 0
        for move in moves:
            i = move[0]
            j = move[1]
            if counter > -1:
                counter += 1
                newBoard[i][j] = pieceType
                value = self.max_value(alpha, beta, depth - 1, 3 - pieceType, newBoard, n + 1)
                if bestValue > value[0]:
                    bestValue = value[0]
                    x_pos = i
                    y_pos = j
                beta = min(bestValue, beta)
                newBoard[i][j] = 0
                if beta <= alpha:
                    return bestValue, x_pos, y_pos
        return bestValue, x_pos, y_pos

    def max_value(self, alpha, beta, depth, pieceType, board, n):
        myPieces = len(self.find_died_pieces(pieceType, board))
        self.myPiecesDied += myPieces
        diedPiece, newBoard = self.removediedpieces(pieceType, deepcopy(board))
        self.diedPiece = diedPiece
        if self.isGameEnd(n):
            self.myPiecesDied -= myPieces
            if self.amIWinner(newBoard):
                return 40, None, None
            else:
                return -20, None, None

        if depth == 0:
            value = self.evaluationFunction(newBoard), 0, 0
            self.myPiecesDied -= myPieces
            return value

        x_pos = None
        y_pos = None
        bestValue = -100000000
        counter = 0
        allMoves = self.getPossibleMove(newBoard, pieceType, 'max')
        for move in allMoves:
            i = move[0]
            j = move[1]
            if True:
                counter += 1
                newBoard[i][j] = pieceType
                self.lastMove = (i, j)
                value = self.min_value(alpha, beta, depth - 1, 3 - pieceType, newBoard, n + 1)

                if bestValue < value[0]:
                    bestValue = value[0]
                    x_pos = i
                    y_pos = j
                alpha = max(bestValue, alpha)
                newBoard[i][j] = 0
                if beta <= alpha:
                    return bestValue, x_pos, y_pos
        return bestValue, x_pos, y_pos

    def alphaBetaStart(self, moveNo):
        alpha, x_pos, y_pos = self.max_value(-100000000, 100000000, 2, piece_type, board, moveNo)
        return x_pos, y_pos

    def get_all_neighbours(self, i, j, board):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def get_connected_neighbour_stone(self, i, j, board):
        neighbors = self.get_all_neighbours(i, j, board)
        same_stones_nearby = []
        for x, y in neighbors:
            if board[x][y] == board[i][j]:
                same_stones_nearby.append((x, y))
        return same_stones_nearby

    def get_all_connected_stone(self, i, j, board):
        stack = [(i, j)]
        explored_connected_stones = []
        while stack:
            x, y = stack.pop()
            explored_connected_stones.append((x, y))
            neigbour_connected = self.get_connected_neighbour_stone(x, y, board)
            for stone in neigbour_connected:
                if stone not in stack and stone not in explored_connected_stones:
                    stack.append(stone)
        return explored_connected_stones

    def is_there_liberty(self, i, j, board):
        connected_stone = self.get_all_connected_stone(i, j, board)
        for x, y in connected_stone:
            neighbors = self.get_all_neighbours(x, y, board)
            for x, y in neighbors:
                if board[x][y] == 0:
                    return True
        return False

    def find_died_pieces(self, piece_type, board):
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == piece_type:
                    if not self.is_there_liberty(i, j, board):
                        died_pieces.append((i, j))
        return died_pieces

    def removediedpieces(self, piece_type, board):

        died_pieces = self.find_died_pieces(piece_type, board)
        if not died_pieces:
            return [], board
        updateBoard = self.remove_pieces(died_pieces, board)
        return died_pieces, updateBoard

    def remove_pieces(self, positions, board):
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        return board


def getLastMove(previousBoard, board, pieceType):
    if pieceType == 1:
        for i in range(5):
            for j in range(5):
                if board[i][j] == 3 - pieceType and previousBoard[i][j] != board[i][j]:
                    return i, j
    return 2, 2


def getLastDiedPieces(previousBoard, board, pieceType):
    died = []
    for i in range(5):
        for j in range(5):
            if previousBoard[i][j] == pieceType and board[i][j] == 0:
                died.append((i, j))
    return died


def getInputParameter(n, path='input.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])
        previous_board = [[0 for i in range(5)] for j in range(5)]
        board = [[0 for i in range(5)] for j in range(5)]

        for row in range(1, n + 1):
            for column in range(5):
                previous_board[row - 1][column] = int(lines[row][column])

        for row in range(n + 1, 2 * n + 1):
            for column in range(5):
                board[row - n - 1][column] = int(lines[row][column])

        return piece_type, previous_board, board


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def getMoves():
    with open('init/move_track.json', 'r') as f:
        file = json.load(f)
    return file


def updateMove(data):
    with open('init/move_track.json', 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = getInputParameter(N)
    dict = getMoves()

    if piece_type == dict["piece_type"]:
        dict["move_no"] = int(dict["move_no"]) + 2
    else:
        dict["piece_type"] = int(piece_type)
        dict["move_no"] = int(piece_type) + 2

    move_no = int(dict["move_no"]) - 2
    opponentLastMove = getLastMove(previous_board, board, piece_type)
    lastDiedPieces = getLastDiedPieces(previous_board, board, piece_type)
    player = AlphaBeta(board, previous_board, opponentLastMove, lastDiedPieces, piece_type)
    possibleAction = player.alphaBetaStart(move_no)

    updateMove(dict)

    action = "PASS"
    if possibleAction[0] is not None:
        action = possibleAction
    print(action)
    writeOutput(action)
