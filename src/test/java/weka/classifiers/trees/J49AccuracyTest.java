package weka.classifiers.trees;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;

import org.junit.Assert;
import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffLoader;

//import org.openjdk.jol.info.GraphLayout;

public class J49AccuracyTest {
    double epsilon = 0.001;
    int minNumObjects = 10;

    @Test
    public void WEKADataTest() throws Exception {
        J49 j491 = new J49();
        J49 j492 = new J49();
        J49 j493 = new J49();
        J48 j48 = new J48();

        URL u;
        DataSource source;
        Instances trainingData;

        String[] fileNames = {"/segment-challenge.arff", "/weather.nominal.arff" , "/segment-test.arff",
                "/soybean.arff", "/supermarket.arff", "/unbalanced.arff", "/vote.arff", "/weather.numeric.arff"};

        for (String file : fileNames) {
            u = J49AccuracyTest.class.getResource(file);
            source = new DataSource(u.getFile());
            trainingData = source.getDataSet();
            trainingData.setClassIndex(trainingData.numAttributes() - 1);

            j491.buildClassifier(trainingData, 1);
            j492.buildClassifier(trainingData, 2);
            j493.buildClassifier(trainingData, 3);
            j48.buildClassifier(trainingData);

            for (int i = 0; i < trainingData.numInstances(); i++) {
                Assert.assertEquals("J49 version 1 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j491.classifyInstance(trainingData.instance(i)), epsilon);
                Assert.assertEquals("J49 version 2 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j492.classifyInstance(trainingData.instance(i)), epsilon);
                Assert.assertEquals("J49 version 3 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j493.classifyInstance(trainingData.instance(i)), epsilon);
            }

        }

    }


    @Test
    public void ServiceTrackingRawDataTimeTest() throws Exception {
        J49 j491 = new J49();
        J49 j492 = new J49();
        J49 j493 = new J49();
        J48 j48 = new J48();
        j491.setMinNumObj(minNumObjects);
        j492.setMinNumObj(minNumObjects);
        j493.setMinNumObj(minNumObjects);
        j48.setMinNumObj(minNumObjects);
        String file = "/allfilters_removed.arff";

        URL u = J49AccuracyTest.class.getResource(file);
        DataSource source = new DataSource(u.getFile());
        Instances trainingData = source.getDataSet();
        trainingData = new Instances(trainingData, 0, 5000);
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        long start;
        long end;

        start = System.nanoTime();
        j48.buildClassifier(trainingData);
        end = System.nanoTime();
        System.out.println("Time to build J48 is:  " + (end - start) + "ns and size is: "  /*GraphLayout.parseInstance(j48).totalSize() */ );

        for(int j = 1; j<4; j++) {
            switch(j) {
                case 1:
                    start = System.nanoTime();
                    j491.buildClassifier(trainingData, 1);
                    end = System.nanoTime();
                    System.out.println("Time to build J491 is: " + (end - start) + "ns and density is: " + j491.getModelDensity() + "%" /*GraphLayout.parseInstance(j491).totalSize()*/ );

                    for (int i = 0; i < 100; i++) {
                        Assert.assertEquals("J491 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j491.classifyInstance(trainingData.instance(i)), epsilon);
                    }
                    break;
                case 2:
                    start = System.nanoTime();
                    j492.buildClassifier(trainingData, 1);
                    end = System.nanoTime();
                    System.out.println("Time to build J492 is: " + (end - start) + "ns and density is: " + j492.getModelDensity() + "%" /*GraphLayout.parseInstance(j491).totalSize()*/ );

                    for (int i = 0; i < 100; i++) {
                        Assert.assertEquals("J492 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j492.classifyInstance(trainingData.instance(i)), epsilon);
                    }
                    break;
                case 3:
                    start = System.nanoTime();
                    j493.buildClassifier(trainingData, 3);
                    end = System.nanoTime();
                    System.out.println("Time to build J493 is: " + (end - start) + "ns and density is: " + j493.getModelDensity() + "%" /*GraphLayout.parseInstance(j492).totalSize()*/);

                    for (int i = 0; i < 100; i++) {
                        Assert.assertEquals("J493 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j493.classifyInstance(trainingData.instance(i)), epsilon);
                    }
                    break;
                default:
            }

            System.gc();
        }

    }

    @Test
    public void ServiceTrackingHashedDataTimeTest() throws Exception {
        J49 j491 = new J49();
        J49 j492 = new J49();
        J48 j48 = new J48();

        j491.setMinNumObj(minNumObjects);
        j492.setMinNumObj(minNumObjects);
        j48.setMinNumObj(minNumObjects);

        String file = "/fullyHashed_32cols.arff";

        URL u = J49AccuracyTest.class.getResource(file);
        DataSource source = new DataSource(u.getFile());
        Instances trainingData = new Instances(source.getDataSet(), 0, 5000);
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        long start;
        long end;

        start = System.nanoTime();
        j491.buildClassifier(trainingData, 1);
        end = System.nanoTime();
        System.out.println("Time to build J491 is: " + (end - start) + "ns and density is: " + j491.getModelDensity() + "%" /*GraphLayout.parseInstance(j491).totalSize() */);

        start = System.nanoTime();
        j492.buildClassifier(trainingData, 2);
        end = System.nanoTime();
        System.out.println("Time to build J492 is: " + (end - start) + "ns and density is: " + j492.getModelDensity() + "%" /*GraphLayout.parseInstance(j492).totalSize()*/);

        start = System.nanoTime();
        j48.buildClassifier(trainingData);
        end = System.nanoTime();
        System.out.println("Time to build J48 is:  " + (end - start) + "ns" /*GraphLayout.parseInstance(j48).totalSize()*/);

        for (int i = 0; i < 100; i++) {
            Assert.assertEquals("J491 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j491.classifyInstance(trainingData.instance(i)), epsilon);
            Assert.assertEquals("J492 classification of " + i, j48.classifyInstance(trainingData.instance(i)), j492.classifyInstance(trainingData.instance(i)), epsilon);

        }

    }

    // Test used to collect data comparing speed and size of different models (J48 vs J491 vs J492)
    /*@Test
    public void ServiceTrackingSerializationDataSizeTest() throws Exception {
        int[] INSTANCE_NUM_TESTS = {1000,2000,3000,4000,5000,10000,20000,40000,80000};  // {1000,2000,3000,4000,5000,10000,20000,40000,80000}
        String[] TEST_VERSIONS = { "J492"}; // {"J48", "J491", "J492"}
        String[] HASH_TYPES = {"Hash", "No Hash"};

        for (int i = 0; i < INSTANCE_NUM_TESTS.length; i++) {
            int NUM_INSTANCES = INSTANCE_NUM_TESTS[i];
            for (int v = 0; v < TEST_VERSIONS.length; v++) {
                String TestVer = TEST_VERSIONS[v];  //  "J48" or "J491" or "J492";
                for (int h = 0; h < HASH_TYPES.length; h++) {
                    String HashType = HASH_TYPES[h]; //  "Hash" or "No Hash

                    Instances trainingData;
                    String outDir;

                    //sets output directory and loads data
                    switch (HashType) {
                        case "Hash":
                            outDir = "/Users/Michael.Frew/Documents/J49SerializationTests/Hashed";

                            String fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/fullyHashed_32cols.arff";
                            BufferedReader reader = new BufferedReader(new FileReader(fileName));
                            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
                            Instances data = arff.getData();
                            trainingData = new Instances(data, 0, NUM_INSTANCES);
                            trainingData.setClassIndex(trainingData.numAttributes() - 1);
                            break;

                        case "No Hash":
                            outDir = "/Users/Michael.Frew/Documents/J49SerializationTests/Not Hashed";

                            fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/allfilters_removed.arff";
                            reader = new BufferedReader(new FileReader(fileName));
                            arff = new ArffLoader.ArffReader(reader);
                            data = arff.getData();
                            trainingData = new Instances(data, 0, NUM_INSTANCES);
                            trainingData.setClassIndex(trainingData.numAttributes() - 1);
                            break;

                        default:
                            outDir = "/Users/Michael.Frew/Documents/J49SerializationTests/Not Hashed";

                            fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/allfilters_removed.arff";
                            reader = new BufferedReader(new FileReader(fileName));
                            arff = new ArffLoader.ArffReader(reader);
                            data = arff.getData();
                            trainingData = new Instances(data, 0, NUM_INSTANCES);
                            trainingData.setClassIndex(trainingData.numAttributes() - 1);
                            break;
                    }

                    String outFile;
                    J49 model = new J49();
                    model.setMinNumObj(minNumObjects);

                    switch (TestVer) {
                        case "J48":
                            //do J48 stuff
                            outFile = outDir + "/J48/" + NUM_INSTANCES + ".ser";
                            model.buildClassifier(trainingData, 0);
                            //System.out.println(model.getModelDensity());
                            break;
                        case "J491":
                            //do J491 stuff
                            outFile = outDir + "/J49 Array of FastUtil/" + NUM_INSTANCES + ".ser";
                            model.buildClassifier(trainingData, 1);
                            //System.out.println(model.getModelDensity());
                            break;
                        case "J492":
                            //do J492 stuff
                            outFile = outDir + "/J49 Maps of Maps/" + NUM_INSTANCES + ".ser";
                            model.buildClassifier(trainingData, 2);
                            //System.out.println(model.getModelDensity());
                            break;
                        case "J493":
                            //do J493 stuff --EJML matrix is not serializable
                            *//*outFile = outDir + "/J49 EJML/"+NUM_INSTANCES + ".ser";
                            model.buildClassifier(trainingData, 3);
                            break;*//*
                        default:
                            //this is temporary
                            outFile = outDir + "/J48/" + NUM_INSTANCES + ".ser";
                            model.buildClassifier(trainingData, 0);
                            //System.out.println(model.getModelDensity());
                            break;

                    }

                    FileOutputStream fos = new FileOutputStream(outFile);
                    ObjectOutputStream out = new ObjectOutputStream(fos);
                    out.writeObject(model);
                    out.close();
                    fos.close();

                    System.gc();
                }
            }
        }

    }*/


    //This test is for calculated both the time to build models of different numbers of instances as well as
    // to measure the memory taken up by these models using the org.openjdk.jol.info.GraphLayout
    /*@Test
    public void ServiceTrackingInMemorySizeTest() throws Exception {
        int[] INSTANCE_NUM_TESTS = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 80000};  // {1000,2000,3000,4000,5000,10000,20000,40000,80000}
        String[] TEST_VERSIONS = {"J491", "J492"}; // { "J491", "J492"} - J48 use up too much memory
        String[] HASH_TYPES = {"Hash", "No Hash"};

        for (int h = 0; h < HASH_TYPES.length; h++) {
            String HashType = HASH_TYPES[h]; //  "Hash" or "No Hash
            System.out.print("Hash Type: " + HashType + " | ");
            for (int v = 0; v < TEST_VERSIONS.length; v++) {
                String TestVer = TEST_VERSIONS[v];  //  "J48" or "J491" or "J492";
                System.out.println("Test Version: " + TestVer + "------------------------------");
                for (int i = 0; i < INSTANCE_NUM_TESTS.length; i++) {
                    int NUM_INSTANCES = INSTANCE_NUM_TESTS[i];

                    Instances trainingData;

                    //sets output directory and loads data
                    switch (HashType) {
                        case "Hash":
                            String fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/fullyHashed_32cols.arff";
                            BufferedReader reader = new BufferedReader(new FileReader(fileName));
                            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
                            Instances data = arff.getData();
                            trainingData = new Instances(data, 0, NUM_INSTANCES);
                            trainingData.setClassIndex(trainingData.numAttributes() - 1);
                            break;

                        case "No Hash":
                            fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/allfilters_removed.arff";
                            reader = new BufferedReader(new FileReader(fileName));
                            arff = new ArffLoader.ArffReader(reader);
                            data = arff.getData();
                            trainingData = new Instances(data, 0, NUM_INSTANCES);
                            trainingData.setClassIndex(trainingData.numAttributes() - 1);
                            break;

                        default:
                            fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/allfilters_removed.arff";
                            reader = new BufferedReader(new FileReader(fileName));
                            arff = new ArffLoader.ArffReader(reader);
                            data = arff.getData();
                            trainingData = new Instances(data, 0, NUM_INSTANCES);
                            trainingData.setClassIndex(trainingData.numAttributes() - 1);
                            break;
                    }

                    J49 model = new J49();
                    long startTime;
                    long endTime;

                    switch (TestVer) {
                        case "J48":
                            //do J48 stuff
                            startTime = System.nanoTime();
                            model.buildClassifier(trainingData);
                            endTime = System.nanoTime();
                            break;
                        case "J491":
                            //do J491 stuff
                            startTime = System.nanoTime();
                            model.buildClassifier(trainingData, 1);
                            endTime = System.nanoTime();
                            break;
                        case "J492":
                            //do J492 stuff
                            startTime = System.nanoTime();
                            model.buildClassifier(trainingData, 2);
                            endTime = System.nanoTime();
                            break;
                        default:
                            //this is temporary
                            startTime = System.nanoTime();
                            model.buildClassifier(trainingData);
                            endTime = System.nanoTime();
                            break;

                    }
                    //to print the speed of building model
                    //System.out.print((endTime - startTime) + ",");

                    //to print the size of the model
                    //System.out.print(GraphLayout.parseInstance(model).totalSize() + ",");

                    System.gc();
                }
                System.out.println();
            }
        }

    }*/

    // Test to compare the classification speed between models
    /*@Test
    public void ServiceTrackingClassificationSpeedTest() throws Exception {
        //order is: Hash: J48, J491, J492; Not Hashed: J48, J491, J492
        String[] HASH_MODELS = {"/Users/Michael.Frew/Documents/J49SerializationTests/Hashed/J48/80000.ser",
                "/Users/Michael.Frew/Documents/J49SerializationTests/Hashed/J49 Array of FastUtil/80000.ser",
                "/Users/Michael.Frew/Documents/J49SerializationTests/Hashed/J49 Maps of Maps/80000.ser"};
        String[] UNHASHED_MODELS = {"/Users/Michael.Frew/Documents/J49SerializationTests/Not Hashed/J48/5000.ser",
                "/Users/Michael.Frew/Documents/J49SerializationTests/Not Hashed/J49 Array of FastUtil/5000.ser",
                "/Users/Michael.Frew/Documents/J49SerializationTests/Not Hashed/J49 Maps of Maps/5000.ser"};

        int NUM_INSTANCES = 1000;
        long startTime;
        long endTime;
        J49 model;
        long total;
        FileInputStream fis;
        ObjectInputStream in;
        double dummy;

        //HASHED data tests
        String fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/fullyHashed_32cols.arff";
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        Instances test = new Instances(data, 3999, NUM_INSTANCES);
        test.setClassIndex(test.numAttributes() - 1);

        for (int m = 0; m < HASH_MODELS.length; m++) {
            String modelPath = HASH_MODELS[m];
            fis = new FileInputStream(modelPath);
            in = new ObjectInputStream(fis);
            model = (J49) in.readObject();
            total = 0;
            for (int i = 0; i < NUM_INSTANCES; i++) {
                startTime = System.nanoTime();
                dummy = model.classifyInstance(test.instance(i));
                endTime = System.nanoTime();
                total += (endTime - startTime);
            }
            System.out.println("Hash test " + m + " time: " + (total / NUM_INSTANCES) + " ns");
        }

        //RAW data tests
        fileName = "/Users/Michael.Frew/Documents/Service Tracking Data prep/GoodFiles/1DayFilter/allfilters_removed.arff";
        reader = new BufferedReader(new FileReader(fileName));
        arff = new ArffLoader.ArffReader(reader);

        for (int m = 0; m < UNHASHED_MODELS.length; m++) {
            String modelPath = HASH_MODELS[m];
            fis = new FileInputStream(modelPath);
            in = new ObjectInputStream(fis);
            model = (J49) in.readObject();
            total = 0;
            for (int i = 0; i < NUM_INSTANCES; i++) {
                startTime = System.nanoTime();
                dummy = model.classifyInstance(test.instance(i));
                endTime = System.nanoTime();
                total += (endTime - startTime);
            }
            System.out.println("No Hash test " + m + " time: " + (total / NUM_INSTANCES) + " ns");
        }

    }*/
}
