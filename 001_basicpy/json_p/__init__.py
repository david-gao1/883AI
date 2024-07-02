import json

if __name__ == '__main__':
    # 写入文件
    numbers = [2, 3, 5, 7, 11, 13]
    filename = 'numbers.json'
    with open(filename, 'w') as f:
        json.dump(numbers, f)
    # 读文件
    with open(filename) as f:
        numbers_read = json.load(f)
        print(numbers_read)
