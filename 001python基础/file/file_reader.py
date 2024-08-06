if __name__ == '__main__':
    with open('pi_digits.txt') as file_object:
        for line in file_object:
            print(line.strip())
