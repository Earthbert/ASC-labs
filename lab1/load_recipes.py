"""
This module contains function to read recipes from files located in the
recipes/ directory.
"""

from os import listdir
from os.path import isfile, join

RECIPES_FOLDER = "recipes"

def read_recipes() -> dict:
    """
    Read recipes from files located in the recipes/ directory and return a dictionary of recipes.

    Returns:
        dict: A dictionary containing the recipes, where the coffee name is the key
              and the value is another dictionary containing the resources and their percentages.
    """
    recipe_files = [f for f in listdir(RECIPES_FOLDER) if isfile(join(RECIPES_FOLDER, f))]

    recipes = {}
    for recipe_file in recipe_files:
        with open(join(RECIPES_FOLDER, recipe_file), "r", encoding='utf8') as file:
            coffee_name = file.readline().strip()
            recipes[coffee_name] = {}
            for line in file:
                resource, percentage = line.strip().split("=")
                recipes[coffee_name][resource] = int(percentage)

    return recipes
