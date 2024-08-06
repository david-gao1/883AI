def make_pizza(size, *toppings):
    """概述要制作的比萨。"""
    print(f"\nMaking a {size}-inch pizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping}")



# Making a 16-inch pizza with the following toppings:
# - pepperoni
#
# Making a 12-inch pizza with the following toppings:
# - mushrooms
# - green peppers
# - extra cheese