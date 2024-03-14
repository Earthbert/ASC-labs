"""
Coffee Factory: A multiple producer - multiple consumer approach

Generate a base class Coffee which knows only the coffee name
Create the Espresso, Americano and Cappuccino classes which inherit the base class knowing that
each coffee type has a predetermined size.
Each of these classes have a get message method

Create 3 additional classes as following:
    * Distributor - A shared space where the producers puts coffees and the consumers takes them
    * CoffeeFactory - An infinite loop, which always sends coffees to the distributor
    * User - Another infinite loop, which always takes coffees from the distributor

The scope of this exercise is to correctly use threads, classes and synchronization objects.
The size of the coffee (ex. small, medium, large) is chosen randomly everytime.
The coffee type is chosen randomly everytime.

Example of output:

Consumer 65 consumed espresso
Factory 7 produced a nice small espresso
Consumer 87 consumed cappuccino
Factory 9 produced an italian medium cappuccino
Consumer 90 consumed americano
Consumer 84 consumed espresso
Factory 8 produced a strong medium americano
Consumer 135 consumed cappuccino
Consumer 94 consumed americano
"""
import random
from time import sleep
from threading import Thread
from threading import Semaphore
from random import random as rand

COFFEE_SIZE = ['small', 'medium', 'large']
NR_FACTORIES = 5
NR_USERS = 5
BUFFER_SIZE = 10

class Coffee:
    """ Base class """
    def __init__(self, name : str, size : str):
        self.name = name
        self.size = size

    def get_name(self) -> str:
        return self.name

    def get_size(self) -> str:
        return self.size


class Espresso(Coffee):
    """ Espresso implementation """
    def __init__(self, size):
        super().__init__('espresso', size)
        self.message = f"Espresso {size}"

    def get_message(self):
        return self.message
    
class Americano (Coffee):
    """ Americano  implementation """
    def __init__(self, size):
        super().__init__('americano', size)
        self.message = f"Americano {size}"

    def get_message(self):
        return self.message
    
class Cappuccino(Coffee):
    """ Cappuccino implementation """
    def __init__(self, size):
        super().__init__('cappuccino', size)
        self.message = f"Cappuccino {size}"

    def get_message(self):
        return self.message

class Distributor:
    """ Distributor class """
    
    def __init__(self, buffer_size : int):
        self.buffer = []
        self.empty_sem : Semaphore = Semaphore(buffer_size)
        self.full_sem : Semaphore = Semaphore(0)
        self.user_mutex : Semaphore = Semaphore(1)
        self.factory_mutex : Semaphore = Semaphore(1)

    def get_coffee(self, user_id : int):
        self.full_sem.acquire()
        self.user_mutex.acquire()
        coffee = self.buffer.pop(0)
        print(f"Consumer {user_id} consumed {coffee.get_message()}")
        self.user_mutex.release()
        self.empty_sem.release()
        
    def put_coffee(self, coffee : Coffee, factory_id : int):
        self.empty_sem.acquire()
        self.factory_mutex.acquire()
        self.buffer.append(coffee)
        print(f"Factory {factory_id} produced a nice {coffee.get_size()} {coffee.get_name()}")
        self.factory_mutex.release()
        self.full_sem.release()

class CoffeeFactory(Thread):
    """ Coffee factory """
    def __init__(self, id : int, distributor : Distributor):
        Thread.__init__(self)
        self.id = id
        self.distributor = distributor

    def run(self):
        while True:
            coffee = self.get_random_coffee()
            self.distributor.put_coffee(coffee, self.id)
            
    def get_random_coffee(self) -> Coffee:
        sleep(random.random())
        return random.choice([Espresso(random.choice(COFFEE_SIZE)), \
                                        Americano(random.choice(COFFEE_SIZE)), \
                                        Cappuccino(random.choice(COFFEE_SIZE))])

class User(Thread):
    """ User class """
    def __init__(self, id : int, distributor : Distributor):
        Thread.__init__(self)
        self.id = id
        self.distributor = distributor

    def run(self):
        while True:
            sleep(random.random())
            self.distributor.get_coffee(self.id)

if __name__ == '__main__':
    distributor = Distributor(BUFFER_SIZE)
    factory_threads = [CoffeeFactory(i, distributor) for i in range(NR_FACTORIES)]
    user_threads = [User(i, distributor) for i in range(NR_USERS)]

    for factory in factory_threads:
        factory.start()
    for user in user_threads:
        user.start()

    for factory in factory_threads:
        factory.join()
    for user in user_threads:
        user.join()
