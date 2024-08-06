import json

if __name__ == '__main__':
    # 如果以前存储了用户名，就加载它。
    # 否则，提示用户输入用户名并存储它。
    filename = 'username.json'
    try:
        with open(filename) as f:
            username = json.load(f)
    except FileNotFoundError:
        username = input("What is your name? ")
        with open(filename, 'w') as f:
            json.dump(username, f)
            print(f"We'll remember you when you come back, {username}!")
    else:
        print(f"Welcome back, {username}!")
