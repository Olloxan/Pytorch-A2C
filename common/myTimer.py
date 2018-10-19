import time

class myTimer():

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.timediff = 0
        self.total_time = 0
        self.last_time = 0
        self.online_mean = 0
        self.online_count = 0

    def update(self, current_time):
        self.timediff = current_time - self.last_time
        if self.last_time > 0:           
            self.total_time += self.timediff
            self.updateAvg()

        self.last_time = current_time

    def getTimeDiff(self):
        return self.timediff

    def getAvgTimeDiff(self):
        return self.online_mean
    
    def updateAvg(self):
        self.online_count += 1
        self.online_mean += (self.timediff - self.online_mean) / self.online_count           

    def getTotalTime(self):
        return self.total_time

    def getTimeToGo(self, loopsToGo):
        return loopsToGo * self.online_mean