package weka.classifiers.trees;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Enumeration;

import org.ejml.data.DMatrixSparseTriplet;
import org.junit.Assert;
import org.junit.Test;

import weka.classifiers.trees.j49.ArrayAndFastUtilDistribution;
import weka.classifiers.trees.j49.C45Split;
import weka.classifiers.trees.j49.Distribution;
import weka.classifiers.trees.j49.DistributionFactory;
import weka.classifiers.trees.j49.EJMLDistribution;
import weka.classifiers.trees.j49.FastUtilDistribution;
import weka.classifiers.trees.j49.OriginalDistribution;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class J49DistributionTest {

    public void compareModels(OriginalDistribution original, Distribution dist) throws Exception {
        double[][] moriginal, mOther;
        //Assert.assertEquals("check min num objects", original.check(2), dists[i].check(2) );

        //compare a matrix
        moriginal = original.matrix();
        mOther = dist.matrix();
        for (int b = 0; b < moriginal.length; b++) {
            for (int c = 0; c < moriginal[b].length; c++) {
                Assert.assertEquals(" | Element [" + b + "][" + c + "]", moriginal[b][c], mOther[b][c], 0.01);
            }
        }
        Assert.assertEquals("max bag", original.maxBag(), dist.maxBag());
        Assert.assertEquals("max class", original.maxClass(), dist.maxClass());
        for (int b = 0; b < original.numBags(); b++) {
            Assert.assertEquals("max class (bag " + b + ")", original.maxClass(b), dist.maxClass(b));
            Assert.assertEquals("num correct (bag " + b + ")", original.numCorrect(b), dist.numCorrect(b), 0.001);
            Assert.assertEquals("perBag", original.perBag(b), dist.perBag(b), 0.001);

            for (int c = 0; c < original.numClasses(); c++) {
                Assert.assertEquals("perClassPerBag(" + b + "," + c + ")", original.perClassPerBag(b, c), dist.perClassPerBag(b, c), 0.001);
                Assert.assertEquals("Laplace prob per class and bag", original.laplaceProb(c, b), dist.laplaceProb(c, b), 0.001);
                Assert.assertEquals("prob(" + c + "," + b + ")", original.prob(c, b), dist.prob(c, b), 0.001);
            }

        }

        for (int c = 0; c < original.numClasses(); c++) {
            Assert.assertEquals("perClass", original.perClass(c), dist.perClass(c), 0.001);
            Assert.assertEquals("Laplace prob", original.laplaceProb(c), dist.laplaceProb(c), 0.001);
            Assert.assertEquals(" prob", original.prob(c), dist.prob(c), 0.001);

        }

        Assert.assertEquals("num correct", original.numCorrect(), dist.numCorrect(), 0.001);
        Assert.assertEquals("num correct", original.numCorrect(0), dist.numCorrect(0), 0.001);

        Assert.assertEquals("numBags", original.numBags(), dist.numBags(), 0.001);
        Assert.assertEquals("numClasses", original.numClasses(), dist.numClasses(), 0.001);
        Assert.assertEquals("getDensity", original.getDensity(), dist.getDensity(), 0.001);
        Assert.assertEquals("num Incorrect", original.numIncorrect(), dist.numIncorrect(), 0.001);

        /*System.out.println("Dumping distributions: ");
        System.out.println("original:");
        System.out.print(original.dumpDistribution());
        System.out.println("other:");
        System.out.print(dist.dumpDistribution());*/

        Assert.assertEquals("dump distribution", original.dumpDistribution(), dist.dumpDistribution());

    }

    @Test
    public void tempTest() throws Exception {
        DMatrixSparseTriplet m = new DMatrixSparseTriplet(3,3,9);

        m.addItem(0,0,20);
        m.addItem(2,1,10);
        m.set(1,1,5);
        m.set(0,0,10);

        System.out.println(m.get(0,0));
        System.out.println(m.get(1,1));
    }

    //uses iris dataset to check FastUtil Distribution and see which type of constructor causes issues
    @Test
    public void J49DistributionComparisonTest() throws Exception {
        String fileName = "/Users/Michael.Frew/Documents/weka-3-8-5/data/weather.nominal.arff";
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances trainingData = arff.getData();
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        DistributionFactory distributionFactory = new DistributionFactory(1);
        C45Split modelToUse = new C45Split(0, 2, trainingData.sumOfWeights(), true, distributionFactory);
        modelToUse.buildClassifier(trainingData);

        OriginalDistribution[] originals = new OriginalDistribution[6];
        FastUtilDistribution[] fastD = new FastUtilDistribution[originals.length];
        ArrayAndFastUtilDistribution[] fastAD = new ArrayAndFastUtilDistribution[originals.length];
        EJMLDistribution[] ejmlD = new EJMLDistribution[originals.length];

        originals[0] = new OriginalDistribution(trainingData);
        fastD[0] = new FastUtilDistribution(trainingData);
        fastAD[0] = new ArrayAndFastUtilDistribution(trainingData);
        ejmlD[0] = new EJMLDistribution(trainingData);

        originals[1] = new OriginalDistribution(trainingData, modelToUse);
        fastD[1] = new FastUtilDistribution(trainingData, modelToUse);
        fastAD[1] = new ArrayAndFastUtilDistribution(trainingData, modelToUse);
        ejmlD[1] = new EJMLDistribution(trainingData, modelToUse);

        Distribution toMerge = originals[1];
        int index = 2;

        originals[2] = new OriginalDistribution(toMerge);
        fastD[2] = new FastUtilDistribution(toMerge);
        fastAD[2] = new ArrayAndFastUtilDistribution(toMerge);
        ejmlD[2] = new EJMLDistribution(toMerge);

        originals[3] = new OriginalDistribution(toMerge, index);
        fastD[3] = new FastUtilDistribution(toMerge, index);
        fastAD[3] = new ArrayAndFastUtilDistribution(toMerge, index);
        ejmlD[3] = new EJMLDistribution(toMerge, index);

        double[][] table = toMerge.matrix();
        int numBags = table.length;
        int numClasses = table[0].length;

        originals[4] = new OriginalDistribution(numBags, numClasses);
        fastD[4] = new FastUtilDistribution(numBags, numClasses);
        fastAD[4] = new ArrayAndFastUtilDistribution(numBags, numClasses);
        ejmlD[4] = new EJMLDistribution(numBags, numClasses);

        Enumeration<Instance> enu = trainingData.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance instance = enu.nextElement();
            originals[4].add(modelToUse.whichSubset(instance), instance);
            fastD[4].add(modelToUse.whichSubset(instance), instance);
            fastAD[4].add(modelToUse.whichSubset(instance), instance);
            ejmlD[4].add(modelToUse.whichSubset(instance), instance);
        }

        originals[5] = new OriginalDistribution(table);
        fastD[5] = new FastUtilDistribution(table);
        fastAD[5] = new ArrayAndFastUtilDistribution(table);
        ejmlD[5] = new EJMLDistribution(table);

        System.out.println("models built");

        for (int i = 0; i < originals.length; i++) {
            compareModels(originals[i], fastD[i]);
            compareModels(originals[i], fastAD[i]);
            compareModels(originals[i], ejmlD[i]);
            System.out.println("Test " + i + " has passed successfully");
        }

    }

}
