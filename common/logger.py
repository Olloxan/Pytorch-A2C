import io
import pickle
import os
import torch

class Logger():
    
    def log(self, data, directoryPath="", filename="dummy"):
        if not os.path.exists(directoryPath):
            os.makedirs(directoryPath)
        
        if not directoryPath.endswith("/"):
            directoryPath += "/"
        directoryPath += filename
        self.dump(data=data, filename=directoryPath)
        print("{0} saved".format(filename))

    def log_state_dict(self, state_dict, path, name=""):
        torch.save(state_dict, path)
        print("Model {0} saved".format(name))

    def dump(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def loadwithoutfirstelem(self, filename):
        data = self.load(filename)
        del data[0]
        return data

    def printDayFormat(self, message, seconds):
        days, hours, minutes, seconds = self.convertSeconds(seconds)
        print(message, '{0:02}:{1:02}:{2:02}:{3:02}'.format(int(days), int(hours), int(minutes), int(seconds)))

    def convertSeconds(self, seconds):
        days = seconds // 86400
        seconds -= (days * 86400)
        hours = seconds//3600
        seconds -= hours * 3600
        minutes = seconds//60
        seconds -= minutes * 60
        return days, hours, minutes, seconds