from threading import enumerate, Event, Thread, Condition

class Master(Thread):
    def __init__(self, max_work : int, work_signal : Condition) -> None:
        Thread.__init__(self, name = "Master")
        self.max_work = max_work
        self.work_signal = work_signal
    
    def set_worker(self, worker : 'Worker'):
        self.worker = worker
    
    def run(self):
        for i in range(self.max_work):
            # generate work
            self.work = i  
            
            with self.work_signal:
            # notify worker
                self.work_signal.notify()                
                # get result
                self.work_signal.wait()
                if self.get_work() + 1 != self.worker.get_result():
                    print ("oops")
                print ("%d -> %d" % (self.work, self.worker.get_result()))
    
    def get_work(self):
        return self.work

class Worker(Thread):
    def __init__(self, terminate : Event, work_signal : Condition) -> None:
        Thread.__init__(self, name = "Worker")
        self.terminate = terminate
        self.work_signal = work_signal

    def set_master(self, master : Master):
        self.master = master
    
    def run(self):
        while(True):
            # wait work
            with self.work_signal:
                self.work_signal.wait()
                if(terminate.is_set()): break
                # generate result
                self.result = self.master.get_work() + 1
                # notify master
                self.work_signal.notify()

    
    def get_result(self):
        return self.result

if __name__ ==  "__main__":
    # create shared objects
    terminate = Event()
    work_signal = Condition()
    
    # start worker and master
    w = Worker(terminate, work_signal)
    m = Master(1000, work_signal)
    w.set_master(m)
    m.set_worker(w)
    w.start()
    m.start()

    # wait for master
    m.join()

    # wait for worker
    with work_signal:
        work_signal.notify()
    terminate.set()
    w.join()

    # print running threads for verification
    print(enumerate())

