import cv2
import threading
import sys


class RTSCapture(cv2.VideoCapture):
    _cur_frame = None
    _reading = False
    schemes = ["rtsp://","rtmp://"]
    @staticmethod
    def create(url, *schemes):
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            pass
        return rtscap

    def isStarted(self):
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()


# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print("usage:")
#         print("need rtsp://xxx")
#         sys.exit()
#
#     rtscap = RTSCapture.create(sys.argv[1])
#     rtscap.start_read()
#
#     while rtscap.isStarted():
#         ok, frame = rtscap.read_latest_frame()
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break
#         if not ok:
#             continue
#
#
#         # inhere
#         cv2.imshow("cam", frame)
#
#
#     rtscap.stop_read()
#     rtscap.release()
#     cv2.destroyAllWindows()

