package weka.classifiers.trees;

import java.net.URL;
import org.junit.Assert;
import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DensityTest {

    @Test
    public void WeatherDataDensityTest() throws Exception {
        J49 j48 = new J49();  // j48 is a J49 object but implements the standard distribution
        J49 j491 = new J49(); // Array of Hashmaps
        J49 j492 = new J49(); // Hashmaps of Hashmaps

        URL u = J49AccuracyTest.class.getResource("/weather.nominal.arff");
        DataSource source = new DataSource(u.getFile());
        Instances trainingData = source.getDataSet();
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        j48.buildClassifier(trainingData, 0);
        j491.buildClassifier(trainingData, 1);
        j492.buildClassifier(trainingData, 2);

        Assert.assertEquals("Density of j48 tree", 54.17, j48.getModelDensity(), 0.01);
        Assert.assertEquals("Density of j491 tree", 54.17, j491.getModelDensity(), 0.01);
        Assert.assertEquals("Density of j492 tree", 54.17, j492.getModelDensity(), 0.01);
    }

}
