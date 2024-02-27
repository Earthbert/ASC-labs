from os import listdir
from os.path import isfile, join

"""
	Bonus task: load all the available coffee recipes from the folder 'recipes/'
	File format:
		first line: coffee name
		next lines: resource=percentage

	info and examples for handling files:
		http://cs.curs.pub.ro/wiki/asc/asc:lab1:index#operatii_cu_fisiere
		https://docs.python.org/3/library/io.html
		https://docs.python.org/3/library/os.path.html
"""

RECIPES_FOLDER = "recipes"

def read_recipes() -> dict:
	recipe_files = [f for f in listdir(RECIPES_FOLDER) if isfile(join(RECIPES_FOLDER, f))]
	
	recipes = {}
	for recipe_file in recipe_files:
		with open(join(RECIPES_FOLDER, recipe_file), "r") as file:
			coffee_name = file.readline().strip()
			recipes[coffee_name] = {}
			for line in file:
				resource, percentage = line.strip().split("=")
				recipes[coffee_name][resource] = int(percentage)
	
	return recipes

