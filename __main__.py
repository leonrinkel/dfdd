import os
import sys
import logging
import argparse
from multiprocessing import Queue, Process

from dataset_loader import load_dataset
from my_types import DatasetItem, WarpTask
from log_handler import log_handler_fn
from warp_worker import warp_worker_fn

# just constants for now, might become args in the future
NUMBER_OF_FRAMES = 100
WARP_RELATIVE_PADDING = 0.1
NUMBER_OF_WARP_WORKERS = 1
WARP_LOG_FILE_NAME = "warp.log"
WARP_PREDICTIONS_FILE_NAME = "warp.json"

def create_warp_task(
    dataset_path: str,
    output_path: str,
    dataset_item: DatasetItem,
) -> WarpTask:
    task = WarpTask()
    task["input_file_path"] = dataset_item["file_path"]
    task["number_of_frames"] = NUMBER_OF_FRAMES
    task["relative_padding"] = WARP_RELATIVE_PADDING
    task["log_file_name"] = WARP_LOG_FILE_NAME
    task["predictions_file_name"] = WARP_PREDICTIONS_FILE_NAME

    # take file input path...
    temp = dataset_item["file_path"]
    # ...strip extension...
    temp = os.path.splitext(temp)[0]
    # ...strip dataset path...
    temp = os.path.relpath(temp, dataset_path)
    # ...prefix output path...
    temp = os.path.join(output_path, temp)
    # ...append suffix
    task["output_path"] = temp + "_warp"

    return task

def main():
    logging.basicConfig()
    logger = logging.getLogger("main")

    # parse cli arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-path", type=str, required=True)
    argparser.add_argument("--annotations-path", type=str, required=True)
    argparser.add_argument("--output-path", type=str, required=True)
    args = argparser.parse_args()
    dataset_path: str = args.dataset_path
    annotations_path: str = args.annotations_path
    output_path: str = args.output_path

    # check whether paths exist
    if not os.path.isdir(dataset_path):
        logger.fatal(
            "invalid dataset path, no such directory: %s",
            dataset_path)
        sys.exit(1)
    if not os.path.isfile(annotations_path):
        logger.fatal(
            "invalid annotations path, no such file: %s",
            annotations_path)
        sys.exit(1)
    if not os.path.isdir(output_path):
        logger.fatal(
            "invalid output path, no such directory: %s",
            output_path)
        sys.exit(1)

    # load dataset and enqueue tasks
    warp_task_queue = Queue()
    dataset = load_dataset(dataset_path, annotations_path)
    for item in dataset:
        warp_task_queue.put(
            create_warp_task(dataset_path, output_path, item))
        break

    # enqueue end of queue markers
    warp_task_queue.put(None)

    # log queue and handler for multiprocess logging
    log_queue = Queue()
    log_handler_process = Process(
        target=log_handler_fn, args=(log_queue,))
    log_handler_process.start()

    def spawn_worker(
        target,
        worker_index: int,
        task_queue: Queue,
    ) -> Process:
        worker_process = Process(
            target=target,
            args=(worker_index, task_queue, log_queue,))
        worker_process.start()
        return worker_process

    worker_processes: list[Process] = []

    # spawn warp workers
    worker_processes.extend([
        spawn_worker(warp_worker_fn, i, warp_task_queue)
        for i in range(NUMBER_OF_WARP_WORKERS)
    ])

    # wait for workers to finish
    for worker_process in worker_processes:
        worker_process.join()

    # signal end of queue
    log_queue.put(None)
    # wait for log handler to finish
    log_handler_process.join()

if __name__ == "__main__":
    main()
