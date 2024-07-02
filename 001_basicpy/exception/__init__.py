if __name__ == '__main__':

    first_name=input("请输入第一个字符")
    second_number=input("请输入第二个字符")
    while True:
        if second_number =='q':
            break
        try:
            answer = int(first_name)/int(second_number)
            print(answer)
        except ZeroDivisionError:
            print("You can't divide by zero!")


