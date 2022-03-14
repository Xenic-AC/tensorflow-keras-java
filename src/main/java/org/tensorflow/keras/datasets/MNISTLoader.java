package org.tensorflow.keras.datasets;

import org.tensorflow.Tensors;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.data.GraphModeTensorFrame;
import org.tensorflow.keras.utils.DataUtils;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.utils.Pair;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

/**
 * Code based on example found at: -
 * https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
 */
public class MNISTLoader {
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABELS_MAGIC = 2049;
    private static final int OUTPUT_CLASSES = 10;

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final String ORIGIN_BASE = "http://yann.lecun.com/exdb/mnist/";

    private static final String LOCAL_PREFIX = "datasets/mnist/";

    public static void download() throws IOException {
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES
        );
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS
        );
        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES,
                "beb4b4806386107117295b2e3e08b4c16a6dfb4f001bfeb97bf25425ba1e08e4", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS,
                "986c5b8cbc6074861436f5581f7798be35c7c0025262d33b4df4c9ef668ec773", DataUtils.Checksum.sha256);
    }

    public static Pair<GraphLoader<Float>, GraphLoader<Float>> graphDataLoader() throws IOException {
        // Download MNIST files if they don't exist.
        MNISTLoader.download();

        float[][] trainImages = readImages(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString());
        float[][] trainLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString());
        float[][] testImages = readImages(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString());
        float[][] testLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString());

        return new Pair<>(
                new GraphModeTensorFrame<>(
                        Float.class, Tensors.create(trainImages), Tensors.create(trainLabels)),
                new GraphModeTensorFrame<>(
                        Float.class, Tensors.create(testImages), Tensors.create(testLabels)));
    }

    private static float[][] readImages(String imagesPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();
            int cols = inputStream.readInt();

            return readImageBuffer(inputStream, numImages, rows * cols);
        }
    }

    private static float[][] readLabelsOneHot(String labelsPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {
            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();
            return readLabelBuffer(inputStream, numLabels);
        }
    }

    private static byte[][] readBatchedBytes(
            DataInputStream inputStream, int batches, int bytesPerBatch) throws IOException {
        byte[][] entries = new byte[batches][bytesPerBatch];
        for (int i = 0; i < batches; i++) {
            inputStream.readFully(entries[i]);
        }
        return entries;
    }

    private static float[][] readImageBuffer(
            DataInputStream inputStream, int numImages, int imageSize) throws IOException {
        byte[][] entries = readBatchedBytes(inputStream, numImages, imageSize);
        float[][] unsignedEntries = new float[numImages][imageSize];
        for (int i = 0; i < unsignedEntries.length; i++) {
            for (int j = 0; j < unsignedEntries[0].length; j++) {
                unsignedEntries[i][j] = (float) (entries[i][j] & 0xFF) / 255.0f;
            }
        }

        return unsignedEntries;
    }

    private static float[][] readLabelBuffer(DataInputStream inputStream, int numLabels)
            throws IOException {
        byte[][] entries = readBatchedBytes(inputStream, numLabels, 1);

        float[][] labels = new float[numLabels][OUTPUT_CLASSES];
        for (int i = 0; i < entries.length; i++) {
            labelToOneHotVector(entries[i][0] & 0xFF, labels[i], false);
        }

        return labels;
    }

    private static void labelToOneHotVector(int label, float[] oneHot, boolean fill) {
        if (label >= oneHot.length) {
            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
        }

        if (fill) Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
    }
}