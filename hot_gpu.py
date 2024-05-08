import tensorflow as tf
import threading
import time

def matrix_operation(gpu_id, matrix_size=9999):
    # 设置当前线程的GPU
    with tf.device(f'/gpu:{gpu_id}'):
        # 创建随机矩阵
        A = tf.random_normal([matrix_size, matrix_size], mean=0.0, stddev=1.0)
        B = tf.random_normal([matrix_size, matrix_size], mean=0.0, stddev=1.0)
        C = tf.matmul(A, B)

    # 在session中运行计算
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        while True:
            # 执行矩阵乘法并计算结果
            result = sess.run(tf.reduce_sum(C))
            # print(f"GPU {gpu_id} computed sum: {result}")
            # time.sleep(0.5)  # 可以根据需要调整延时

def main():
    # 启动两个线程，分别在两个GPU上运行
    threads = [
        threading.Thread(target=matrix_operation, args=(0,))  # 第一个GPU
        # threading.Thread(target=matrix_operation, args=(1,))   # 第二个GPU
    ]

    # 启动线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
