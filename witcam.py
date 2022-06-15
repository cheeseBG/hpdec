import multiprocessing
from multiprocessing import Process, Queue, Pipe
from server.rtClassifier_server import realTimeHPDec
from yolo.Pedestrian_detection import yolo_detect

q = Queue()

if __name__ == "__main__":
    parent, child = Pipe()
    th1 = Process(target=realTimeHPDec, args=(child,))
    th2 = Process(target=yolo_detect, args=(parent,))

    th1.start()
    th2.start()
    th1.join()
    th2.join()
