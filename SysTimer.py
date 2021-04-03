import time

class Timer(object):

    def __init__(self, text):
        self.text = text
        self.start = time.process_time()

    def __enter__(self):
        return
    
    def __exit__(self, type, value, traceback):
        stop = time.process_time()
        self.delta = stop - self.start
        attributes = {
            "milliseconds": self.delta * 1000,
            "seconds": self.delta,
            "minutes": self.delta / 60,
        }
        text = self.text.format(self.delta, **attributes)
        
        print(text)

