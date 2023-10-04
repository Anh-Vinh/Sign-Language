input = []

with open('input.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        input.append(list(line.split()))

def valid_col(input, col):
    for num in range(1, 10):
        if sum(row[col] == num for row in input) != 1:
            return False
    return True

def valid_row(input, row):
    for num in range(1, 10):
        if input[row].count(num) != 1:
            return False
    return True

def valid_box(input, start_row, start_col):
    for num in range(1, 10):
        if sum(row[start_col:start_col+3].count(num) for row in input[start_row:start_row+3]) != 1:
            return False
    return True

print(input)