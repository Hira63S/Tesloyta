# Multithreading intro notebook
# separate flow of execution.

import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    # to call one thread to finish, but wait for the next one to finish and not kill the program
    # we call x.join()
    logging.info("Main      :  before creating thread")
    x = threading.Thread(target=thread_function, args=(1,), daemon=True) # creating and # pass in 1 as the argument
    logging.info("Main      :  before running thread")
    x.start()                                               # starting thread
    logging.info("Main      : wait for the thread to finish")
    x.join()
    logging.info("Main      : all done")
