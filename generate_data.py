import math
import copy
import random
import pandas as pd

class Player:
    def __init__(self):
        self.white = 10
        self.black = 10

class Board:
    def __init__(self):
        self.turn = 1
        self.hero = Player()
        self.villain = Player()
        self.move_number = 0
        self.board_size = 6
        self.stone_count = 0
        self.board_array = [(0,0)]* (self.board_size**2)
        self.winner = None
        self.index_map()

    def display(self):
        rowlen = self.board_size
        for r in range(rowlen):
            string = ''
            for c in range(rowlen):
                char = self.board_array[r*6 + c][0]
                if char == -1:
                    char = 'B'
                elif char == 0:
                    char = '-'
                elif char == 1:
                    char = 'W'

                string += char
                while len(string) < 5*(c+1):
                    string += ' '
            print(string)
        print("Hero: " + str(self.hero.white) + 'w ' + str(self.hero.black) + 'b')
        print("Villain: " + str(self.villain.white) + 'w ' + str(self.villain.black) + 'b')
        print('Move Number: ' + str(self.move_number))

    def index_map(self):
        rowlen = self.board_size
        index = rowlen*math.floor(rowlen/2) + math.floor((rowlen/2)-1)
        direction = ['r', 'u', 'l', 'd']
        d = 0
        scale = 1
        count = 0
        for i in range(rowlen**2):
            self.board_array[index] = (0, i)
            if direction[d] == 'r':
                index += 1
            elif direction[d] == 'u':
                index -= rowlen
            elif direction[d] == 'l':
                index -= 1
            elif direction[d] == 'd':
                index += rowlen
            count += 1
            if count == scale:
                d = (d+1)%4
            if count == 2*scale:
                scale += 1
                count = 0
                d = (d+1)%4

    def get_move_location(self):
        position = self.board_size**2
        index = None
        for i in range(self.board_size**2):
            if self.board_array[i][0] == 0:
                if self.board_array[i][1] <= position:
                    position = self.board_array[i][1]
                    index = i
        return index

    def remove(self, position):
        self.board_array[position] = (0, self.board_array[position][1])
        self.stone_count -= 1
    def add_stone(self, position):
        color = self.board_array[position][0]
        if self.turn == 1:
            if color == 1:
                self.hero.white += 1
            if color == -1:
                self.hero.black += 1
        if self.turn == -1:
            if color == 1:
                self.villain.white += 1
            if color == -1:
                self.villain.black += 1
    def lose_stone(self, color):
        if self.turn == 1:
            if color == 1:
                self.hero.white -= 1
            if color == -1:
                self.hero.black -= 1
        if self.turn == -1:
            if color == 1:
                self.villain.white -= 1
            if color == -1:
                self.villain.black -= 1

    def translate(self, color):
        if color in ['B', 'b', 'black', 'Black']:
            result = -1
        elif color in ['W', 'w', 'white', 'White']:
            result = 1
        return result
    def capture(self, position):
        color = self.board_array[position][0]
        #upper left
        bool = (position%self.board_size >= 2)\
            and (position//self.board_size >= 2)\
            and (self.board_array[position - 2 - 2 * self.board_size][0] == color)\
            and (self.board_array[position - 1 - self.board_size][0] == -1 * color)
        if bool:
            p = position - 1 - self.board_size
            self.add_stone(p)
            self.remove(p)

       #upper
        bool = (position//self.board_size >= 2)\
            and (self.board_array[position - 2 * self.board_size][0] == color)\
            and (self.board_array[position - self.board_size][0] == -1 * color)
        if bool:
            p = position - self.board_size
            self.add_stone(p)
            self.remove(p)

        #upper right
        bool = (position%self.board_size <= self.board_size-3)\
            and (position//self.board_size >= 2)\
            and (self.board_array[position + 2 - 2 * self.board_size][0] == color)\
            and (self.board_array[position + 1 - self.board_size][0] == -1 * color)
        if bool:
            p = position + 1 - self.board_size
            self.add_stone(p)
            self.remove(p)

        #left
        bool = (position%self.board_size >= 2)\
            and (self.board_array[position - 2][0] == color)\
            and (self.board_array[position - 1][0] == -1 * color)
        if bool:
            p = position - 1
            self.add_stone(p)
            self.remove(p)

        #right
        bool = (position%self.board_size <= self.board_size - 3)\
            and (self.board_array[position + 2][0] == color)\
            and (self.board_array[position + 1][0] == -1 * color)
        if bool:
            p = position + 1
            self.add_stone(p)
            self.remove(p)

        #lower left
        bool = (position%self.board_size >= 2)\
            and (position//self.board_size <= self.board_size - 3)\
            and (self.board_array[position - 2 + 2 * self.board_size][0] == color)\
            and (self.board_array[position - 1 + self.board_size][0] == -1 * color)
        if bool:
            p = position - 1 + self.board_size
            self.add_stone(p)
            self.remove(p)

        #lower
        bool = (position//self.board_size <= self.board_size - 3)\
            and (self.board_array[position + 2 * self.board_size][0] == color)\
            and (self.board_array[position + self.board_size][0] == -1 * color)
        if bool:
            p = position + self.board_size
            self.add_stone(p)
            self.remove(p)

        #lower right
        bool = (position%self.board_size <= self.board_size - 3)\
            and (position//self.board_size <= self.board_size - 3)\
            and (self.board_array[position + 2 + 2 * self.board_size][0] == color)\
            and (self.board_array[position + 1 + self.board_size][0] == -1 * color)
        if bool:
            p = position + 1 + self.board_size
            self.add_stone(p)
            self.remove(p)

    def play(self, move):
        m = self.translate(move)
        position = self.get_move_location()
        self.board_array[position] = (m, self.board_array[position][1])
        self.lose_stone(m)
        self.capture(position)
        self.turn *= -1
        self.move_number += 1
        self.stone_count += 1
        self.check_win_state()

    def check_win_state(self):
        if self.stone_count == self.board_size**2:
            if self.hero.white + self.hero.black > self.villain.white + self.villain.black:
                self.winner = 1
            elif self.hero.white + self.hero.black < self.villain.white + self.villain.black:
                self.winner = -1
            else:
                self.winner = 0
        if self.move_number > 100:
            self.winner = 0
        if self.hero.white + self.hero.black <= 0:
            self.winner = -1
        if self.villain.white + self.villain.black <= 0:
            self.winner = 1


def evaluate(board, depth = 15):
    if board.move_number > 100:
        return 0.5
    if board.hero.white < 0 or board.hero.black < 0:
        return 0
    if board.villain.white < 0 or board.villain.black < 0:
        return 1
    if board.winner == 1:
        return 1
    if board.winner == -1:
        return 0
    if board.winner == 0:
        return 0.5

    if depth == 0:
        hero = board.hero.white + board.hero.black
        villain = board.villain.white + board.villain.black
        return hero / (hero + villain)
    else:
        cpy = copy.deepcopy(board)
        cpy.play('w')
        white_score = evaluate(cpy, depth - 1)

        cpy = copy.deepcopy(board)
        cpy.play('b')
        black_score = evaluate(cpy, depth - 1)

        if board.turn == 1:
            return max(white_score, black_score)
        if board.turn == -1:
            return min(white_score, black_score)

def make_move(board, strength = 3, eval_depth = 15, deterministic=False):
    move = None
    if board.move_number == 0:
        board.play('w')
        move = 'W'
        white_score = 0.5
        black_score = 0.5
    elif board.turn == 1 and board.hero.white == 0 and board.hero.black > 0:
        board.play('b')
        black_score = evaluate(board, depth=eval_depth)
        white_score = None
        move = 'B'
    elif board.turn == 1 and board.hero.white > 0 and board.hero.black == 0:
        board.play('w')
        white_score = evaluate(board, depth=eval_depth)
        black_score = None
        move = 'W'
    elif board.turn == -1 and board.villain.white == 0 and board.villain.black > 0:
        board.play('b')
        black_score = evaluate(board, depth=eval_depth)
        white_score = None
        move = 'B'
    elif board.turn == -1 and board.villain.white > 0 and board.villain.black == 0:
        board.play('w')
        white_score = evaluate(board, depth=eval_depth)
        black_score = None
        move = 'W'
    else:
        temp = copy.deepcopy(board)
        temp.play('w')
        white_score = evaluate(temp, depth=eval_depth)
        temp = copy.deepcopy(board)
        temp.play('b')
        black_score = evaluate(temp, depth=eval_depth)

        if white_score == black_score:
            p_white = 0.5
        elif deterministic == True:
            if board.turn == 1:
                if white_score > black_score:
                    p_white = 1
                else:
                    p_white = 0
            if board.turn == -1:
                if white_score > black_score:
                    p_white = 0
                else:
                    p_white = 1
        else:
            if board.turn == 1:
                p_white = white_score**strength / (white_score**strength + black_score**strength)
            if board.turn == -1:
                p_white = (1-white_score)**strength / ((1-white_score)**strength + (1-black_score)**strength)

        if random.random() < p_white:
            board.play('w')
            move = 'W'
        else:
            board.play('b')
            move = 'B'
    board.display()
    return {'eval': [white_score, black_score], 'move': move}

def play_game():
    strength = 3
    eval_depth = 15

    board = Board()
    board.play('w')
    fs_data = pd.read_csv('feng_shuei_data.csv')
    game_num = max(fs_data['game_number']) + 1
    new_data = new_data_framework()
    print('New Game: #' + str(game_num))
    while board.winner == None:
        move_info = make_move(board, strength, eval_depth)
        append_board_state(new_data, board)
        new_data['player_strength'].append(strength)
        new_data['evaluation_depth'].append(eval_depth)
        new_data['played_move'].append(move_info['move'])
        new_data['white_eval'].append(move_info['eval'][0])
        new_data['black_eval'].append(move_info['eval'][1])
        new_data['game_number'].append(game_num)
    for i in range(1, board.move_number):
        new_data['outcome'].append(board.winner)

    new_data = pd.DataFrame(new_data)
    fs_data = pd.concat([fs_data, new_data])
    fs_data.to_csv('feng_shuei_data.csv', index = False)
    return new_data

def new_data_framework():
    data = {}
    for i in range(36):
        string = 'p_' + str(i)
        data[string] = []
    data['h_white'] = []
    data['h_black'] = []
    data['v_white'] = []
    data['v_black'] = []
    data['turn'] = []
    data['move_number'] = []
    data['player_strength'] = []
    data['evaluation_depth'] = []
    data['outcome'] = []
    data['played_move'] = []
    data['white_eval'] = []
    data['black_eval'] = []
    data['game_number'] = []

    return data

def append_board_state(data_framework, board):
    for i in range(len(board.board_array)):
        string = 'p_' + str(board.board_array[i][1])
        data_framework[string].append(board.board_array[i][0])
    data_framework['h_white'].append(board.hero.white)
    data_framework['h_black'].append(board.hero.black)
    data_framework['v_white'].append(board.villain.white)
    data_framework['v_black'].append(board.villain.black)
    data_framework['turn'].append(board.turn)
    data_framework['move_number'].append(board.move_number)