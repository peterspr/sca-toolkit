from src.notscared.file_handling.readh5 import ReadH5
from src.notscared.tasks.Task import Task, Options
from multiprocessing import Process
from multiprocessing import Pool

class NotScared:
    def __init__(self, filename: str, task: Task, task_options: Options, tile: tuple):
        self.task_options = task_options
        self.filename = filename

        self.tiles = tile
        
        self.results = [[None for _ in range(self.tiles[1])] for _ in range(self.tiles[0])]
        # self.threads = calculate_threads
        self.tasks = [[task(task_options) for _ in range(self.tiles[1])] for _ in range(self.tiles[0])]
        
    def multi_process_run(self, tile):
        reader = ReadH5(self.filename, tile)
        task = self.tasks[tile[0]][tile[1]]
        # push
        while reader.next():
            # get data
            task.push(reader.get_batch_samples(), reader.get_batch_ptxts())
        
        task.calculate()
        self.results[tile[0]][tile[1]] = task.get_results()

    def run_process_no_pool(self):
        process_array = [[Process(self.multi_process_run((x, y))) for y in range(self.tiles[1])] for x in range(self.tiles[0])]
        for x in range(self.tiles[0]):
            for y in range(self.tiles[1]):
                process_array[x][y].start()

        for x in range(self.tiles[0]):
            for y in range(self.tiles[1]):
                process_array[x][y].join()
                
    def run(self):
        tiles = [(x, y) for x in range(self.tiles[0]) for y in range(self.tiles[1])]
        with Pool() as pool:
            results = pool.map(self.multi_process_run, tiles)
            pool.close()
            pool.join()

        

    def get_results_of_tile(tile):
        pass

    def get_all_results(self):
        return self.results

    def get_heat_map():
        pass

    def collapse_tiles():
        pass
    