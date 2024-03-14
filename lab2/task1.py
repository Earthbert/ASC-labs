"""
    Basic thread handling exercise:

    Use the Thread class to create and run more than 10 threads which print their name and a random
    number they receive as argument. The number of threads must be received from the command line.

    e.g. Hello, I'm Thread-96 and I received the number 42

"""

import random
from threading import Thread
  
class MyThread(Thread):
    """ Clasa care incapsuleaza codul nostru concurent """
    def __init__(self, id, nr):
        Thread.__init__(self)
        self.id = id
        self.nr = nr

    def run(self):
        print (f"Thread-{self.id} says: {self.nr}")

if __name__ == '__main__':
    thread_list = []

    for i in range(100):
        t = MyThread(i, random.randint(0, 100))
        t.start()
        thread_list.append(t)

    print ("Main thread says: hello from main")

    for t in thread_list:
        t.join()
