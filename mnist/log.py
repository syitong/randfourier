import time

class log:
    """
    This module is used to track the progress of events
    and write it into a log file.
    """
    def __init__(self,filepath,init_message):
        self.message = ''
        self.filepath = filepath
        self.progress = {'task':[init_message],'time':[time.process_time()]}
        print(self.progress['task'][-1]
              + ': {:.4f}'.format(self.progress['time'][-1]))

    def time_event(self,message):
        self.progress['task'].append(message)
        self.progress['time'].append(time.process_time())
        print(self.progress['task'][-1]
              + ': {:.4f}'.format(self.progress['time'][-1]
                              - self.progress['time'][-2]))

    def record(self,message):
        self.message = message

    def save(self):
        progress = self.progress
        with open(self.filepath,'w') as logfile:
            for idx in range(1,len(progress['task'])):
                logfile.write(progress['task'][idx]
                              + ': {:.4f}\n'.format(progress['time'][idx]
                                                    - progress['time'][idx - 1]))
            logfile.write(self.message)
