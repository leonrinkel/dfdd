import os
import cv2
import dlib
import logging
import logging.handlers
from multiprocessing import Queue

from my_types import WarpTask

def warp_worker_fn(
    worker_index: int,
    task_queue: Queue,
    log_queue: Queue,
):
    # prepare logger
    logger = logging.getLogger(f"warp_worker_{worker_index}")
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logger.setLevel(logging.DEBUG)
    logger.info("started")

    while True:
        task: WarpTask = task_queue.get()
        if task is None: break

        # create output direvtory if it does not exist already
        if not os.path.isdir(task["output_path"]):
            logger.info(
                "creating output directory: %s",
                task["output_path"])
            os.makedirs(task["output_path"])

        # open input file as video capture
        logger.info(
            "opening video capture for file: %s",
            task["input_file_path"])
        capture = cv2.VideoCapture(task["input_file_path"])
        if not capture.isOpened():
            logger.warning(
                "unable to open video capture for file: %s",
                task["input_file_path"])
            continue

        # create face detector
        detector = dlib.get_frontal_face_detector()

        # read video frame by frame
        read_frame_count = 0
        written_frame_count = 0
        warn_frame_count = 0
        while capture.isOpened():
            # stop after specified nunber of frames
            if read_frame_count == task["number_of_frames"]:
                break

            # read frame
            retval, frame = capture.read()
            if not retval: break
            read_frame_count += 1

            # convert color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # prepare output path
            frame_output_path = os.path.join(
                task["output_path"], f"{read_frame_count}.png")
            if os.path.isfile(frame_output_path):
                logger.debug(
                    "skipping frame %d of %s because output file already exists",
                    read_frame_count, task["input_file_path"])
                continue

            # detect faces
            faces = detector(frame, 1)
            if len(faces) == 0:
                logger.warning(
                    "skipping frame %d of %s because there were no faces",
                    read_frame_count, task["input_file_path"])
                warn_frame_count += 1
                continue
            if len(faces) > 1:
                logger.warning(
                    "skipping frame %d of %s because there were multiple faces",
                    read_frame_count, task["input_file_path"])
                warn_frame_count += 1
                continue

            # crop out face
            height, width = frame.shape[0], frame.shape[1]
            top = int(max(0, faces[0].top() - task["relative_padding"] * height))
            left = int(max(0, faces[0].left() - task["relative_padding"] * width))
            bottom = int(min(height, faces[0].bottom() + task["relative_padding"] * height))
            right = int(min(width, faces[0].right() + task["relative_padding"] * width))
            cropped_frame = frame[top:bottom, left:right]

            # save cropped frame to file
            dlib.save_image(cropped_frame, frame_output_path)
            written_frame_count += 1

        # close video capture
        capture.release()

        # log preprocessing results
        logger.info("read %d frames, wrote %d frames of %s",
            read_frame_count, written_frame_count, task["input_file_path"])

        # check whether to proceed with prediction
        if warn_frame_count > 0:
            logger.warning(
                "not continuing with prediction because there were warnings")
            continue
        log_path = os.path.join(
            task["output_path"], task["log_file_name"])
        predictions_path = os.path.join(
            task["output_path"], task["predictions_file_name"])
        if os.path.isfile(log_path) and os.path.isfile(predictions_path):
            logger.info(
                "skipping prediction because because output files already exist")
            continue

        # build predict command
        data_path = task["output_path"]
        helper_path = os.path.join(os.getcwd(), "helper")
        predictions_file_name = task["predictions_file_name"]
        command = \
            "docker run --rm -it " \
            f"-v {data_path}:/data " \
            f"-v {helper_path}:/helper " \
            "leonrinkel/cvprw2019-face-artifacts " \
            "python /helper/warp_helper.py " \
            f"{predictions_file_name} > {log_path}"

        # run predict command
        logger.debug("executing command: %s", command)
        os.system(command)

        logger.info(
            "finished prediction, approximate tasks remaining: %d",
            task_queue.qsize())

    logger.info("finished")
