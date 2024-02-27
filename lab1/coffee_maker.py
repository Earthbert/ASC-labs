"""
A command-line controlled coffee maker.
"""

import sys
from load_recipes import read_recipes

"""
Implement the coffee maker's commands. Interact with the user via stdin and print to stdout.

Requirements:
    - use functions
    - use __main__ code block
    - access and modify dicts and/or lists
    - use at least once some string formatting (e.g. functions such as strip(), lower(),
    format()) and types of printing (e.g. "%s %s" % tuple(["a", "b"]) prints "a b"
    - BONUS: read the coffee recipes from a file, put the file-handling code in another module
    and import it (see the recipes/ folder)

There's a section in the lab with syntax and examples for each requirement.

Feel free to define more commands, other coffee types, more resources if you'd like and have time.
"""

"""
Tips:
*  Start by showing a message to the user to enter a command, remove our initial messages
*  Keep types of available coffees in a data structure such as a list or dict
e.g. a dict with coffee name as a key and another dict with resource mappings (resource:percent)
as value
"""

# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"  #!!! when making coffee you must first check that you have enough resources!
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

# Coffee maker's resources - the values represent the fill percents

coffee_cost = {}

resources = {}

"""
Example result/interactions:

I'm a smart coffee maker
Enter command:
list
americano, cappuccino, espresso
Enter command:
status
water: 100%
coffee: 100%
milk: 100%
Enter command:
make
Which coffee?
espresso
Here's your espresso!
Enter command:
refill
Which resource? Type 'all' for refilling everything
water
water: 100%
coffee: 90%
milk: 100%
Enter command:
exit
"""

def list_coffee():
    print(", ".join(coffee_cost.keys()))

def resource_status():
    print("\n".join(["%s: %s" % (k, v) for k, v in resources.items()]))

def refill_resource():
    print("Which resource? Type 'all' for refilling everything")
    resource = sys.stdin.readline().strip().lower()
    
    if resource == "all":
        for k in resources:
            resources[k] = 100
    else:
        resources[resource] = 100
    print("\n".join(["%s: %s" % (k, v) for k, v in resources.items()]))

def command_make_coffee():
    print("Which coffee?")
    coffee = sys.stdin.readline().strip().lower()
    
    if (coffee not in coffee_cost):
        print("Unknown coffee type")
        return
    
    if (any(resources[k] < v for k, v in coffee_cost[coffee].items())):
        print("Not enough resources")
        return
    
    for k, v in coffee_cost[coffee].items():
        resources[k] -= v
    
    print("Here's your %s!" % coffee)

def command_help():
    print("Available commands: %s" % ", ".join(commands))

commands_map = {
    LIST_COFFEES: list_coffee,
    RESOURCE_STATUS: resource_status,
    MAKE_COFFEE: command_make_coffee,
    REFILL: refill_resource,
    HELP: command_help
}

if __name__ == "__main__":
    recipes = read_recipes()
    for k, v in recipes.items():
        coffee_cost[k] = v
        for i in v:
            resources[i] = 100

    print("I'm a silly coffee maker")
    print("How can I help you? Type 'help' for available commands")
    while line := sys.stdin.readline():
        command = line.strip().lower()
        if command == EXIT:
            print("Goodbye!")
            break
        elif command in commands_map:
            commands_map[command]()
        else:
            print("Unknown command. Type 'help' for available commands")

