class Dog:
    """一次模拟小狗的简单尝试。"""
    def __init__(self, name, age):
        """初始化属性name和age。"""
        self.name = name
        self.age = age


    def sit(self):
        """模拟小狗收到命令时蹲下。"""
        print(f"{self.name} is now sitting.")


    def roll_over(self):
        """模拟小狗收到命令时打滚。"""
        print(f"{self.name} rolled over!")



class Car:
    """一次模拟汽车的简单尝试。"""
    def __init__(self, make, model, year):
        """初始化描述汽车的属性。"""
        self.make = make
        self.model = model
        self.year = year

    def get_descriptive_name(self):
        """返回整洁的描述性信息。"""
        long_name = f"{self.year} {self.make} {self.model}"
        return long_name.title()




class Car:
    def __init__(self, make, model, year):
        """初始化描述汽车的属性。"""
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 5

    def get_descriptive_name(self):
        """--snip--"""
        print(f"This car has {self.year} year old")


    def read_odometer(self):
        """打印一条指出汽车里程的消息。"""
        print(f"This car has {self.odometer_reading} miles on it.")


    def update_odometer(self, mileage):
        """
        将里程表读数设置为指定的值。
        禁止将里程表读数往回调。
        """
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can't roll back an odometer!")

    def increment_odometer(self, miles):
        """将里程表读数增加指定的量。"""
        self.odometer_reading += miles




if __name__ == '__main__':
    my_used_car = Car('subaru', 'outback', 2015)
    print(my_used_car.get_descriptive_name())

    my_used_car.update_odometer(23_500)
    my_used_car.read_odometer()

    my_used_car.increment_odometer(100)
    my_used_car.read_odometer()

    my_used_car.odometer_reading=1000
    my_used_car.read_odometer()





    # my_new_car = Car('audi', 'a4', 2019)
    # print(my_new_car.get_descriptive_name())
    #
    # my_new_car.update_odometer(23)
    # my_new_car.read_odometer()
    # my_new_car = Car('audi', 'a4', 2019)
    # print(my_new_car.get_descriptive_name())
    #
    # my_new_car.odometer_reading = 23
    # my_new_car.read_odometer()


    # my_new_car = Car('audi', 'a4', 2019)
    # print(my_new_car.get_descriptive_name())
    # my_new_car.read_odometer()


    # my_dog=Dog("Willie",6)
    # print(f"My dog's name is {my_dog.name}.")
    # print(f"My dog is {my_dog.age} years old.")

    # my_new_car = Car('audi', 'a4', 2019)
    # print(my_new_car.get_descriptive_name())
