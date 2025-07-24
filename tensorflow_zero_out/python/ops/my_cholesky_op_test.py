import tensorflow as tf
import my_cholesky_op as myop

def test_basic():
    x = tf.constant([[4.0, 0.0], [2.0, 3.0]], dtype=tf.float32)
    result = myop.my_cholesky_op(x)
    print("Cholesky result:", result)

if __name__ == "__main__":
    test_basic()