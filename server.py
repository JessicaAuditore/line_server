import socket
import threading
import sys
import os
import struct
from detection.line_detection import Line_detection
from recognition.line_recognition import Line_recognition


class Server:
    def __init__(self):
        if not os.path.exists('./input'):
            os.makedirs('./input')
        if not os.path.exists('./output'):
            os.makedirs('./output')
        self.detection_model = Line_detection()
        self.recognition_model = Line_recognition()
        self.open_socket()

    def open_socket(self):
        try:
            s = socket.socket()
            s.bind(("0.0.0.0", 1024))
            s.listen(10)
        except socket.error as msg:
            print(msg)
            sys.exit(1)
        print("Waiting...")

        while 1:
            conn, addr = s.accept()
            conn.send('Connect to the server...'.encode())
            threading.Thread(target=self.handle, args=(conn, addr)).start()

    def handle(self, conn, addr):
        print('Accept a new connection from {}'.format(addr))
        while 1:
            input_path = self.recv_from_client(conn, addr)
            detection_output_path, box_list = self.detection_model.predict(input_path)
            self.send_to_client(conn, detection_output_path, addr)
            recognition_output_path = self.recognition_model.predict(input_path, box_list)
            self.send_to_client(conn, recognition_output_path, addr)

    @staticmethod
    def recv_from_client(conn, addr):
        file_info_size = struct.calcsize('128sl')
        buf = conn.recv(file_info_size)
        if buf:
            file_name, file_size = struct.unpack('128sl', buf)
            path = os.path.join(str.encode('./input'), file_name.strip(str.encode('\00')))
            recv_size = 0
            fp = open(path, 'wb')
            while not recv_size == file_size:
                if file_size - recv_size > 1024:
                    data = conn.recv(1024)
                    recv_size += len(data)
                else:
                    data = conn.recv(file_size - recv_size)
                    recv_size = file_size
                fp.write(data)
            fp.close()
            print('{} recv from {}...'.format(path.decode(), addr))
            return path.decode()

    @staticmethod
    def send_to_client(conn, path, addr):
        file_head = struct.pack('128sl', bytes(os.path.basename(path).encode('utf-8')), os.stat(path).st_size)
        conn.send(file_head)
        fp = open(path, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                print('{} send to {}...'.format(path, addr))
                break
            conn.send(data)


if __name__ == '__main__':
    Server()
