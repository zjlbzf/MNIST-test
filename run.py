import threading
import os


def process():
    os.sys('conda activate py3.7')
    os.sys('G:\code\Projects\AI\Competation\Kaggle\003_MNIST> & D:/ProgramData/Anaconda3/envs/py3.7/python.exe g:/code/Projects/AI/Competation/Kaggle/003_MNIST/tf_MNIST.py')


for i in range(3):
    t1 = threading.Thread(target=process, args=[])
    tt.start()
