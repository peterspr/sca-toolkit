from file_handling.readh5 import ReadH5
from tasks.Task import Task, Options

class NotScared:
    def __init__(self, filename: str, task: Task, task_options: Options, tile: tuple):
        self.task = task(task_options)
        self.task_options = task_options
        self.filename = filename

        self.tiles = tile
        self.reader = ReadH5(self.filename, self.tiles)
        
        self.results = None
        # self.threads = calculate_threads
        self.tasks = None
        

    def run(self):
        # push
        while self.reader.next():
            # get data
            self.task.push(self.reader.get_batch_samples(), self.reader.get_batch_ptxts())
        
        self.task.calculate()
        self.results = self.task.get_results()

    def get_results_of_tile(tile):
        pass

    def get_all_results():
        pass

    def get_heat_map():
        pass

    def collapse_tiles():
        pass
    