from multiprocessing import Process
from multiprocessing import Pool
from .file_handling.readh5 import ReadH5
from .tasks.Task import Task, Options

class NotScared:
    def __init__(self, filename: str, task: Task, task_options: Options):
        self.task_options = task_options
        self.filename = filename

        self.tiles = self.get_num_tiles()

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

    def get_results_of_tile(self, tile):
        pass

    def get_all_results(self):
        return self.results

    def get_heat_map(self):
        pass

    def collapse_tiles(self):
        pass

    def get_num_tiles(self):
        # initialize x and y
        x = 0
        y = 0
        exists = True
        while exists:
            try:
                # access by key base off tiles
                self.filename[f"traces/tile_{x}/"]
                x += 1
            except KeyError:
                exists = False

        exists = True
        while exists:
            try:
                self.filename[f"traces/tile_0/tile_{y}/"]
                y += 1
            except KeyError:
                exists = False
        return (x, y)
    