import logging
import cv2
import numpy as np
import socket
import sys
import time

from senseact.communicator import Communicator

class RealSenseCommunicator(Communicator):
    """
    Intel Real Sense Communicator for interfacing.
    """

    def __init__(self, host='localhost', port=5000, height=480, width=640, num_channels=3):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        self.server_address = (host, port)

        self._height = height
        self._width = width
        self._num_channels = num_channels
        self._packet_size = self._height * self._width * self._num_channels

        self._old_image = np.zeros((self._num_channels, self._height, self._width), dtype=np.uint8)

        sensor_args = {
            'array_len': self._packet_size,
            'array_type': 'd',
            'np_array_type': 'd',
        }

        super(RealSenseCommunicator, self).__init__(use_sensor=True,
                                                    use_actuator=False,
                                                    sensor_args=sensor_args,
                                                    actuator_args={})

    def run(self):
        print('connecting to {} port {}'.format(*self.server_address))
        self.sock.connect(self.server_address)

        super(RealSenseCommunicator, self).run()
        
        self.sock.sendall(b'done')
        self.sock.close()

    def terminate(self):
        self.sock.sendall(b'done')
        self.sock.close()
        super(RealSenseCommunicator, self).terminate()

    def _sensor_handler(self):
        self.sock.sendall(b'get')

        received_data = b''
        while len(received_data) < self._packet_size:
            received_data += self.sock.recv(self._packet_size)

        image = np.frombuffer(received_data, dtype=np.uint8)

        # Check for image change
        if np.array_equal(image, self._old_image):
            return

        self._old_image = image
        self.sensor_buffer.write(image.flatten().astype(float) / 255)
        time.sleep(0.01)

    def _actuator_handler(self):
        """
        There is no actuator for cameras.
        """
        raise RuntimeError("Real Sense Communicator does not have an actuator handler.")


if __name__ == "__main__":
    communicator = RealSenseCommunicator()
    communicator.run()
