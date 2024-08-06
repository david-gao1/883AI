def greet_user():
    """你好这是一个注释文档"""
    print("hello")


def greet_user2(username):
    """显示简单的问候语。"""
    print(f"Hello, {username.title()}!")


def describe_pet(animal_type, pet_name):
    """显示宠物的信息。"""
    print(f"\nI have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name.title()}.")


def describe_pet(pet_name, animal_type='dog'):
    """显示宠物的信息。"""
    print(f"\nI have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name.title()}.")


def get_formatted_name(first_name, last_name):
    """返回名字"""
    full_name = f"{first_name} {last_name}"
    return full_name.title()


def get_formatted_name(first_name, last_name, middle_name=''):
    """返回整洁的姓名。"""
    if middle_name:  # 不为空的判断
        full_name = f"{first_name} {middle_name} {last_name}"
    else:
        full_name = f"{first_name} {last_name}"
    return full_name.title()


def build_person(first_name, last_name):
    """返回一个字典，其中包含有关一个人的信息。"""
    person = {'first': first_name, 'last': last_name}
    return person


def build_person(first_name, last_name, age=None):
    """返回一个字典，其中包含有关一个人的信息。"""
    person = {'first': first_name, 'last': last_name}
    if age:
        person['age'] = age
    return person


def get_formatted_name(first_name, last_name):
    """返回整洁的姓名。"""
    full_name = f"{first_name} {last_name}"
    return full_name.title()


def greet_users(names):
    """向列表中的每位用户发出简单的问候。"""
    for name in names:
        msg = f"Hello, {name.title()}!"
        print(msg)




def print_models(unprinted_designs, completed_models):
    """
    模拟打印每个设计，直到没有未打印的设计为止。
    打印每个设计后，都将其移到列表completed_models中。
    """
    while unprinted_designs:
        current_design = unprinted_designs.pop()
        print(f"Printing model: {current_design}")
        completed_models.append(current_design)


def show_completed_models(completed_models):
    """显示打印好的所有模型。"""
    print("\nThe following models have been printed:")
    for completed_model in completed_models:
        print(completed_model)




def make_pizza(*toppings):
    """打印顾客点的所有配料。"""
    print(toppings)


def make_pizza(size, *toppings):
    """概述要制作的比萨。"""
    print(f"\nMaking a {size}-inch pizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping}")


def build_profile(first, last, **user_info):
    """创建一个字典，其中包含我们知道的有关用户的一切。"""
    user_info['first_name'] = first
    user_info['last_name'] = last
    return user_info



if __name__ == '__main__':
    user_profile = build_profile('albert', 'einstein',
                                 location='princeton',
                                 field='physics')
    print(user_profile)


    # make_pizza(16, 'pepperoni')
    # make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
    #
    # make_pizza('pepperoni')
    # make_pizza('mushrooms', 'green peppers', 'extra cheese')

    # unprinted_designs = ['phone case', 'robot pendant', 'dodecahedron']
    # completed_models = []
    #
    # print_models(unprinted_designs, completed_models)
    # show_completed_models(completed_models)
    # greet_user()
    # greet_user2('jesse')
    #
    # describe_pet(animal_type='hamster', pet_name='harry')
    #
    # describe_pet(pet_name='willie')
    #
    # musician = get_formatted_name('roman', 'gao')
    # print(musician)

    # musician = get_formatted_name('jimi', 'hendrix')
    # print(musician)
    #
    # musician = get_formatted_name('john', 'hooker', 'lee')
    # print(musician)

    # musician = build_person('jimi', 'hendrix')
    # print(musician)

    # musician = build_person('jimi', 'hendrix', age=27)
    # print(musician)

    # 这是一个无限循环
    # while True:
    #     print("\nPlease tell me your name:")
    #     f_name = input("First name: ")
    #     l_name = input("Last name: ")
    #
    #     formatted_name = get_formatted_name(f_name, l_name)
    #     print(f"\nHello, {formatted_name}!")

    # usernames = ['hannah', 'ty', 'margot']
    # greet_users(usernames)
