
from threading import Thread
from threading import Semaphore
from time import sleep

NR_PHILOSOPHERS = 100

class Philosopher(Thread):
    def __init__(self, id : int, left_fork : Semaphore, right_fork : Semaphore):
        Thread.__init__(self)
        self.id = id
        self.left_fork = left_fork
        self.right_fork = right_fork
        
    def run(self):
        if self.id % 2 == 0:
            self.left_fork.acquire()
            self.right_fork.acquire()
        else:
            self.right_fork.acquire()
            self.left_fork.acquire()
            
        print(f"Philosopher {self.name} is eating")
        sleep(0.5)
        
        self.left_fork.release()
        self.right_fork.release()

if __name__ == '__main__':
    forks = [Semaphore(1) for _ in range(NR_PHILOSOPHERS)]
    philosophers = [Philosopher(i, forks[i], forks[(i + 1) % NR_PHILOSOPHERS]) for i in range(NR_PHILOSOPHERS)]
    
    for p in philosophers:
        p.start()
        
    for p in philosophers:
        p.join()
