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
MAKE_COFFEE = "make"
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

coffee_cost = {}

resources = {}

def _list_coffee():
    print(", ".join(coffee_cost.keys()))

def _resource_status():
    print("\n".join([f"{k}: {v}" for k, v in resources.items()]))

def _refill_resource():
    print("Which resource? Type 'all' for refilling everything")
    resource = sys.stdin.readline().strip().lower()

    if resource == "all":
        for k in resources:
            resources[k] = 100
    elif resource not in resources:
        print("Unknown resource")
    else:
        resources[resource] = 100
    print("\n".join([f"{k}: {v}" for k, v in resources.items()]))

def _command_make_coffee():
    print("Which coffee?")
    coffee = sys.stdin.readline().strip().lower()

    if coffee not in coffee_cost:
        print("Unknown coffee type")
        return

    if any(resources[k] < v for k, v in coffee_cost[coffee].items()):
        print("Not enough resources")
        return

    for k, v in coffee_cost[coffee].items():
        resources[k] -= v
    
    print(f"Here's your {coffee}!")

def _command_help():
    print("Available commands: %s" % ", ".join(commands))

commands_map = {
    LIST_COFFEES: _list_coffee,
    RESOURCE_STATUS: _resource_status,
    MAKE_COFFEE: _command_make_coffee,
    REFILL: _refill_resource,
    HELP: _command_help
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
        if command in commands_map:
            commands_map[command]()
        else:
            print("Unknown command. Type 'help' for available commands")
