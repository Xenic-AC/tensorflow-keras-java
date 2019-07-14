package io.gitlab.keras.activations;


import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.math.Sigmoid;
import org.tensorflow.op.math.Tanh;
import org.tensorflow.op.nn.*;


/**
 * Helper functions to compute activations using a TF Ops object.
 */
public enum Activations {
    sigmoid, tanh, relu, elu, selu, softmax, logsoftmax;

    public static <T extends Number> Activation<T> select(String activation) {
        return new Activation<>(Activations.select(Activations.valueOf(activation)));
    }

    public static <T extends Number> Activation<T> select(Activations type) {
        switch (type) {
            case sigmoid:
                return new Activation<T>(Activations::sigmoid);
            case tanh:
                return new Activation<T>(Activations::tanh);
            case relu:
                return new Activation<T>(Activations::relu);
            case elu:
                return new Activation<T>(Activations::elu);
            case selu:
                return new Activation<T>(Activations::selu);
            case softmax:
                return new Activation<T>(Activations::softmax);
            case logsoftmax:
                return new Activation<T>(Activations::logSoftmax);
            default:
                throw new IllegalArgumentException("Invalid ActivationType");
        }
    }
    /**
     * Sigmoid activation function.
     */
    public static <T> Sigmoid<T> sigmoid(Ops tf, Operand<T> x) {
        return tf.math.sigmoid(x);
    }

    /**
     * Tanh activation function.
     */
    public static <T> Tanh<T> tanh(Ops tf, Operand<T> x) {
        return tf.math.tanh(x);
    }

    /**
     * Rectified Linear Unit.
     */
    public static <T> Relu<T> relu(Ops tf, Operand<T> features) {
        return tf.nn.relu(features);
    }

    /**
     * Exponential Linear Unit.
     */
    public static <T extends Number> Elu<T> elu(Ops tf, Operand<T> features) {
        return tf.nn.elu(features);
    }

    /**
     * Scaled Exponential Linear Unit.
     */
    public static <T extends Number> Selu<T> selu(Ops tf, Operand<T> features) {
        return tf.nn.selu(features);
    }

    /**
     * Softmax activation function.
     */
    public static <T extends Number> Softmax<T> softmax(Ops tf, Operand<T> logits) {
        return tf.nn.softmax(logits);
    }

    /**
     * Log-Softmax activation function.
     */
    public static <T extends Number> LogSoftmax<T> logSoftmax(Ops tf, Operand<T> logits) {
        return tf.nn.logSoftmax(logits);
    }
}