"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time
import threading

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


# def run(input_path, output_path, model_path, model_type="./weights/dpt_beit_large_512.pt", optimize=False, side=False, height=None,
#         square=False, grayscale=False):
#     """Run MonoDepthNN to compute depth maps.

#     Args:
#         input_path (str): path to input folder
#         output_path (str): path to output folder
#         model_path (str): path to saved model
#         model_type (str): the model type
#         optimize (bool): optimize the model to half-floats on CUDA?
#         side (bool): RGB and depth side by side in output images?
#         height (int): inference encoder image height
#         square (bool): resize to a square resolution?
#         grayscale (bool): use a grayscale colormap?
#     """
#     print("Initialize")

#     # select device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device: %s" % device)

#     model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

#     # get input
#     if input_path is not None:
#         image_names = glob.glob(os.path.join(input_path, "*"))
#         num_images = len(image_names)
#     else:
#         print("No input path specified. Grabbing images from camera.")

#     # create output folder
#     if output_path is not None:
#         os.makedirs(output_path, exist_ok=True)

#     print("Start processing")

#     if input_path is not None:
#         if output_path is None:
#             print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
#         for index, image_name in enumerate(image_names):

#             print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

#             # input
#             original_image_rgb = utils.read_image(image_name)  # in [0, 1]
#             image = transform({"image": original_image_rgb})["image"]

#             # compute
#             with torch.no_grad():
#                 prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
#                                      optimize, False)

#             # output
#             if output_path is not None:
#                 filename = os.path.join(
#                     output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
#                 )
#                 if not side:
#                     utils.write_depth(filename, prediction, grayscale, bits=2)
#                 else:
#                     original_image_bgr = np.flip(original_image_rgb, 2)
#                     content = create_side_by_side(original_image_bgr*255, prediction, grayscale)
#                     cv2.imwrite(filename + ".png", content)
#                 utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))


import cv2
import os
import torch
import numpy as np
#from depth_utils import load_model, process
#from depth_utils import create_side_by_side, utils

import cv2
import numpy as np
import torch
import os
import utils

def run(video_path, output_path, model_path, model_type="./weights/dpt_beit_large_512.pt", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    """Run MonoDepthNN to compute depth maps from a video file.

    Args:
        video_path (str): path to input video file
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    frame_index = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        print("  Processing frame %d/%d" % (frame_index + 1, frame_count))

        # input
        original_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform({"image": original_image_rgb})["image"]

        # compute
        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                 optimize, False)

        # output
        if output_path is not None:
            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(video_path))[0] + '-frame-' + str(frame_index) + '-' + model_type
            )
            if not side:
                utils.write_depth(filename, prediction, grayscale, bits=2)
            else:
                original_image_bgr = np.flip(original_image_rgb, 2)
                content = create_side_by_side(original_image_bgr*255, prediction, grayscale)
                cv2.imwrite(filename + ".png", content)
            utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))

        # Display original and depth frames side by side
        if side:
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            # Normalize depth values to range [0, 1] for proper visualization
            depth_image = prediction.astype(np.float32)
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image = (depth_image * 255).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            depth_image = cv2.resize(depth_image, (original_image_bgr.shape[1], original_image_bgr.shape[0]))
            combined_image = np.hstack((original_image_bgr, depth_image))
            cv2.imshow("Original vs Depth", combined_image)
        else:
            # Normalize depth values to range [0, 1] for proper visualization
            depth_image = prediction.astype(np.float32)
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image = (depth_image * 255).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_image)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    video_capture.release()
    cv2.destroyAllWindows()
    print("Finished")
#run(video_path="input/video.mp4", output_path="output", model_path="model/model.pt", model_type="large", optimize=False, side=False, height=None, square=False, grayscale=False)



    # else:
    #     with torch.no_grad():
    #         fps = 1
    #         video = VideoStream(0).start()
    #         time_start = time.time()
    #         frame_index = 0
    #         while True:
    #             frame = video.read()
    #             if frame is not None:
    #                 original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
    #                 image = transform({"image": original_image_rgb/255})["image"]

    #                 prediction = process(device, model, model_type, image, (net_w, net_h),
    #                                      original_image_rgb.shape[1::-1], optimize, True)

    #                 original_image_bgr = np.flip(original_image_rgb, 2) if side else None
    #                 content = create_side_by_side(original_image_bgr, prediction, grayscale)
    #                 cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', content/255)

    #                 if output_path is not None:
    #                     filename = os.path.join(output_path, 'Camera' + '-' + model_type + '_' + str(frame_index))
    #                     cv2.imwrite(filename + ".png", content)

    #                 alpha = 0.1
    #                 if time.time()-time_start > 0:
    #                     fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
    #                     time_start = time.time()
    #                 print(f"\rFPS: {round(fps,2)}", end="")

    #                 if cv2.waitKey(1) == 27:  # Escape key
    #                     break

    #                 frame_index += 1
    #     print()

    # print("Finished")


import os
import cv2
import torch
import numpy as np
import argparse
import threading
import time
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
from utils import write_depth, write_pfm, read_image

def process_video(video_path, output_path, model_weights, model_type, optimize, side, height, square, grayscale):
    """Process a video file to compute depth maps."""
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_weights, model_type, optimize, height, square)

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    frame_index = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        print("  Processing frame %d/%d" % (frame_index + 1, frame_count))

        # input
        original_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform({"image": original_image_rgb})["image"]

        # compute
        with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                 optimize, False)

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(video_path))[0] + '-frame-' + str(frame_index) + '-' + model_type
        )
        if not side:
            write_depth(filename, prediction, grayscale, bits=2)
        else:
            original_image_bgr = np.flip(original_image_rgb, 2)
            content = create_side_by_side(original_image_bgr*255, prediction, grayscale)
            cv2.imwrite(filename + ".png", content)
        write_pfm(filename + ".pfm", prediction.astype(np.float32))

        # Display original and depth frames side by side
        if side:
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            # Normalize depth values to range [0, 1] for proper visualization
            depth_image = prediction.astype(np.float32)
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image = (depth_image * 255).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            depth_image = cv2.resize(depth_image, (original_image_bgr.shape[1], original_image_bgr.shape[0]))
            combined_image = np.hstack((original_image_bgr, depth_image))
            cv2.imshow("Original vs Depth", combined_image)
        else:
            # Normalize depth values to range [0, 1] for proper visualization
            depth_image = prediction.astype(np.float32)
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image = (depth_image * 255).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_image)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    video_capture.release()
    cv2.destroyAllWindows()
    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute depth maps for multiple videos")
    parser.add_argument("-i", "--input_paths", nargs="+", type=str, help="List of input video paths")
    parser.add_argument("-o", "--output_path", type=str, help="Path to output folder")
    parser.add_argument("-m", "--model_weights", type=str, help="Path to model weights")
    parser.add_argument("-t", "--model_type", type=str, default="dpt_beit_large_512", help="Model type")
    parser.add_argument("--optimize", action="store_true", help="Optimize the model to half-floats on CUDA")
    parser.add_argument("-s", "--side", action="store_true", help="Show RGB and depth side by side in output images")
    parser.add_argument("--height", type=int, help="Inference encoder image height")
    parser.add_argument("--square", action="store_true", help="Resize to a square resolution")
    parser.add_argument("--grayscale", action="store_true", help="Use a grayscale colormap")
    args = parser.parse_args()

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    threads = []
    for video_path in args.input_paths:
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found.")
            continue

        output_folder = args.output_path if args.output_path else os.path.dirname(video_path)
        thread = threading.Thread(target=process_video, args=(video_path, output_folder, args.model_weights,
                                                              args.model_type, args.optimize, args.side,
                                                              args.height, args.square, args.grayscale))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()