package org.tensorflow.keras.layers;

import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.utils.Keras;

public class Layers {
    // Builders for Input Layer
    public static Input input(long firstDim, long... units) {
        return new Input(Keras.concatenate(firstDim,units));
    }

    // Builders for Dense Layer
    public static Dense dense(int units) {
        return new Dense(units, Dense.options());
    }

    public static Dense dense(int units, Dense.Options options) {
        return new Dense(units, options);
    }

    public static  Dense dense(int units, Activation<Float> activation) {
        return new Dense(units, Dense.Options.builder().setActivation(activation).build());
    }

    public static  Dense dense(int units, Activations activation) {
        return new Dense(units, Dense.Options.builder().setActivation(activation).build());
    }

    public static  Dense dense(int units, Activations activation, Initializers kernelInitializer, Initializers biasInitializer) {
        return new Dense(units, Dense.Options.builder()
                .setActivation(activation)
                .setKernelInitializer(kernelInitializer)
                .setBiasInitializer(biasInitializer)
                .build());
    }

    public static  Dense dense(int units, Activation<Float> activation, Initializer<Float> kernelInitializer, Initializer<Float> biasInitializer) {
        return new Dense(units, Dense.Options.builder()
                .setActivation(activation)
                .setKernelInitializer(kernelInitializer)
                .setBiasInitializer(biasInitializer)
                .build());
    }

    // Builders for Flatten Layer
    public static Flatten flatten() {
        return new Flatten();
    }


    public static class Options<T> {
        Class dtype = Float.class;

        public Builder<T> builder() {
            return new Builder<>();
        }

        public static class Builder<T> {
            Options<T> options;

            public Builder() {
                this.options = new Options<>();
            }

            public Builder<T> dtype(Class<T> dtype) {
                this.options.dtype = dtype;
                return this;
            }

            public Options<T> build() {
                return this.options;
            }
        }
    }
}