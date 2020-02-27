import argparse
import numpy as np
import pyrealsense2 as rs
import socket
import sys

from args.parser import parse_tcp_server_args

def run(args):
    assert args.height > 0
    assert args.width > 0
    assert args.frame_rate > 0
    assert args.colour_format in ("rgb8",)

    # Get colour format
    if args.colour_format == "rgb8":
        colour_format = rs.format.rgb8
    else:
        raise NotImplementedError()

    # Get device serial number
    realsense_ctx = rs.context()
    device_sn = realsense_ctx.devices[args.device_id].get_info(rs.camera_info.serial_number)
    print("ID: {} - S/N: {}".format(args.device_id, device_sn))

    # Intel RealSense pipeline
    config = rs.config()
    config.enable_device(device_sn)
    config.enable_stream(rs.stream.color, args.width, args.height, colour_format, args.frame_rate)

    pipeline = rs.pipeline()
    pipeline.start(config)

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = (args.host, args.port)
    print('Starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    try:
        while True:
            # Wait for a connection
            print('waiting for a connection')
            connection, client_address = sock.accept()
            try:
                print('connection from', client_address)

                # Receive the data in small chunks and retransmit it
                while True:
                    data = connection.recv(16).decode("utf-8")
                    if data == 'done':
                        break

                    if data != 'get':
                        continue

                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        continue

                    # Convert images to numpy arrays
                    # Data Shape: (Width, Height, Channels)
                    image = np.asanyarray(color_frame.get_data()).transpose(2, 0, 1)
                    connection.send(image.tobytes())
            finally:
                print('Closing connection from {}:{}', *client_address)
                # Clean up the connection
                connection.close()
    finally:
        pipeline.stop()
        sock.close()

if __name__ == "__main__":
    args = parse_tcp_server_args()
    run(args)
