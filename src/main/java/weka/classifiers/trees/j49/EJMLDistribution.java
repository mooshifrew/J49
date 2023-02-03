package weka.classifiers.trees.j49;

import java.util.Enumeration;

import org.ejml.data.DMatrixSparseTriplet;

import it.unimi.dsi.fastutil.doubles.Double2DoubleMap;
import it.unimi.dsi.fastutil.doubles.Double2DoubleOpenHashMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

public class EJMLDistribution implements Distribution {

    int INIT_LENGTH = 10;

    //m_perBag uses standard implementation
    double[] m_perBag;

    //m_perClass uses hashmap
    Double2DoubleOpenHashMap m_perClass;

    //use DMatrixSparseTriplet for 2D array
    DMatrixSparseTriplet m_perClassPerBag;

    double totaL;

    int m_numClasses;

    //methods for adding so that the implementation can be changed easily
    private void addToPerBag(int bagIndex, double weight) {
        m_perBag[bagIndex] += weight;
    }

    private void addToPerClass(double classIndex, double weight) {
        m_perClass.addTo(classIndex, weight);
    }

    private void addToPerClassPerBag(int bagIndex, double classVal, double weight) {
        double currWeight = m_perClassPerBag.get(bagIndex, (int) classVal);

        if (currWeight == 0) {
            m_perClassPerBag.addItem(bagIndex, (int) classVal, weight);
        }
        else {
            m_perClassPerBag.set(bagIndex, (int) classVal, currWeight + weight);
        }
    }

    /**
     * Creates and initializes a new distribution.
     */
    public EJMLDistribution(int numBags, int numClasses) {
        m_perBag = new double[numBags];
        m_perClass = new Double2DoubleOpenHashMap();
        m_perClassPerBag = new DMatrixSparseTriplet(numBags, numClasses, INIT_LENGTH);
        totaL = 0;
        m_numClasses = numClasses;
    }

    /**
     * Creates and initializes a new distribution using the given array of format OriginalDistribution.m_perClassPerBag
     */
    public EJMLDistribution(double[][] table) {
        m_perBag = new double[table.length];
        m_perClass = new Double2DoubleOpenHashMap(table[0].length);
        m_perClassPerBag = new DMatrixSparseTriplet(table.length, table[0].length, INIT_LENGTH);
        totaL = 0;
        m_numClasses = table[0].length;

        for (int b = 0; b < table.length; b++) {
            for (int c = 0; c < table[b].length; c++) {
                if (table[b][c] != 0) {
                    addToPerBag(b, table[b][c]);
                    addToPerClass(c, table[b][c]);
                    addToPerClassPerBag(b,c,table[b][c]);
                    totaL += table[b][c];
                }
            }
        }
        m_perClass.trim();
        m_perClassPerBag.shrinkArrays();
    }

    /**
     * Creates a distribution with only one bag according to instances in source.
     *
     * @throws Exception if something goes wrong
     */
    public EJMLDistribution(Instances source) throws Exception {
        m_perBag = new double[1];
        m_perClass = new Double2DoubleOpenHashMap(source.numClasses());
        m_perClassPerBag = new DMatrixSparseTriplet(1, source.numClasses(), INIT_LENGTH);
        totaL = 0;
        m_numClasses = source.numClasses();

        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            add(0, enu.nextElement());
        }

        m_perClass.trim();
        m_perClassPerBag.shrinkArrays();
    }

    /**
     * Creates a distribution according to given instances and split model.
     *
     * @throws Exception if something goes wrong
     */

    public EJMLDistribution(Instances source, ClassifierSplitModel modelToUse) throws Exception {
        int index;
        Instance instance;
        double[] weights;

        totaL = 0;
        m_numClasses = source.numClasses();

        m_perBag = new double[modelToUse.numSubsets()];
        m_perClass = new Double2DoubleOpenHashMap(m_numClasses);
        m_perClassPerBag = new DMatrixSparseTriplet(modelToUse.numSubsets(), numClasses(), INIT_LENGTH);

        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = enu.nextElement();
            index = modelToUse.whichSubset(instance);
            if (index != -1) {
                add(index, instance);
            }
            else {
                weights = modelToUse.weights(instance);
                addWeights(instance, weights);
            }
        }

        m_perClass.trim();
        m_perClassPerBag.shrinkArrays();

    }

    /**
     * Creates distribution with only one bag by merging all bags of given
     * distribution.
     */
    public EJMLDistribution(Distribution toMerge) {
        m_perBag = new double[1];
        m_perBag[0] = toMerge.total();

        m_perClass = new Double2DoubleOpenHashMap(toMerge.numClasses());
        m_perClassPerBag = new DMatrixSparseTriplet(1, toMerge.numClasses(), INIT_LENGTH);

        for (int i = 0; i < toMerge.numClasses(); i++) {
            if (toMerge.perClass(i) != 0) {
                addToPerClass(i, toMerge.perClass(i));
                addToPerClassPerBag(0, i, toMerge.perClass(i));
            }
        }

        totaL = toMerge.total();
        m_numClasses = toMerge.numClasses();

        m_perClass.trim();
        m_perClassPerBag.shrinkArrays();

    }

    /**
     * Creates distribution with two bags by merging all bags apart of the
     * indicated one.
     */
    public EJMLDistribution(Distribution toMerge, int index) {
        int c;
        double weight;

        totaL = toMerge.total();
        m_numClasses = toMerge.numClasses();

        m_perBag = new double[2];
        m_perBag[0] = toMerge.perBag(index);
        m_perBag[1] = toMerge.total() - m_perBag[0];

        m_perClass = new Double2DoubleOpenHashMap(m_numClasses);
        m_perClassPerBag = new DMatrixSparseTriplet(2, numClasses(), INIT_LENGTH);

        for (c = 0; c < toMerge.numClasses(); c++) {
            addToPerClass(c, toMerge.perClass(c));
            addToPerClassPerBag(0,c,toMerge.perClassPerBag(index, c));
            weight = toMerge.perClass(c) - toMerge.perClassPerBag(index, c);
            if (weight != 0) {
                addToPerClassPerBag(1, c, weight);
            }
        }

        m_perClass.trim();
        m_perClassPerBag.shrinkArrays();
    }

    /**
     * Returns number of non-empty bags of distribution.
     */
    @Override
    public int actualNumBags() {
        int count = 0;
        for (int i = 0; i < numBags(); i++) {
            if (perBag(i) != 0) {
                count++;
            }
        }
        return count;
    }

    /**
     * Returns number of classes actually occuring in distribution.
     */
    @Override
    public int actualNumClasses() {
        return m_perClass.size();
    }

    /**
     * Returns number of classes actually occuring in given bag.
     *
     * @param bagIndex
     */
    @Override
    public int actualNumClasses(int bagIndex) {
        int count = 0;
        if (perBag(bagIndex) == 0) {
            return 0;
        }
        else{
            for (int c = 0; c < numClasses(); c++) {
                if (perClass(c) != 0) {
                    if (perClassPerBag(bagIndex, c) != 0) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    /**
     * Adds given instance to given bag.
     *
     * @param bagIndex
     * @param instance
     * @throws Exception if something goes wrong
     */
    @Override
    public void add(int bagIndex, Instance instance) throws Exception {
        double weight = instance.weight();
        double classVal = instance.classValue();

        if (weight == 0) {
            return;
        }

        addToPerBag(bagIndex, weight);
        addToPerClass(classVal, weight);
        addToPerClassPerBag(bagIndex, classVal, weight);
        totaL += weight;
    }

    /**
     * Subtracts given instance from given bag.
     *
     * @param bagIndex
     * @param instance
     * @throws Exception if something goes wrong
     */
    @Override
    public void sub(int bagIndex, Instance instance) throws Exception {
        double weight = instance.weight();
        double classVal = instance.classValue();

        if (weight == 0) {
            return;
        }

        addToPerBag(bagIndex, -weight);
        addToPerClass(classVal, -weight);
        addToPerClassPerBag(bagIndex, classVal, -weight);
        totaL -= weight;
    }

    /**
     * Adds counts to given bag.
     *
     * @param bagIndex
     * @param counts
     */
    @Override
    public void add(int bagIndex, double[] counts) {
        double sum = Utils.sum(counts);
        int c;

        addToPerBag(bagIndex, sum);
        for (c = 0; c < counts.length; c++) {
            if (counts[c] != 0) {
                addToPerClass(c, counts[c]);
                addToPerClassPerBag(bagIndex, c, counts[c]);
            }
        }
        totaL += sum;
    }

    /**
     * Adds all instances with unknown values for given attribute, weighted
     * according to frequency of instances in each bag.
     *
     * @param source
     * @param attIndex
     * @throws Exception if something goes wrong
     */
    @Override
    public void addInstWithUnknown(Instances source, int attIndex) throws Exception {
        double[] probs;
        double weight, newWeight;
        double classIndex;
        Instance instance;
        int b;

        probs = new double[numBags()];
        for (b = 0; b < numBags(); b++) {
            if (Utils.eq(total(), 0)) {
                probs[b] = 1.0 / probs.length;
            }
            else {
                probs[b] = perBag(b) / total();
            }
        }
        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = enu.nextElement();
            if (instance.isMissing(attIndex)) {
                classIndex = instance.classValue();
                weight = instance.weight();
                addToPerClass(classIndex, weight);
                totaL += weight;

                for (b = 0; b < numBags(); b++) {
                    newWeight = probs[b] * weight;
                    addToPerClassPerBag(b, classIndex, newWeight);
                    addToPerBag(b, newWeight);
                }

            }
        }
    }

    /**
     * Adds all instances in given range to given bag.
     *
     * @param bagIndex
     * @param source
     * @param startIndex
     * @param lastPlusOne
     * @throws Exception if something goes wrong
     */
    @Override
    public void addRange(int bagIndex, Instances source, int startIndex, int lastPlusOne) throws Exception {
        double sumOfWeights = 0;
        double classVal;
        double weight;
        int i;

        for (i = startIndex; i < lastPlusOne; i++) {
            weight = source.instance(i).weight();
            if (weight != 0) {
                sumOfWeights += weight;
                classVal = source.instance(i).classValue();
                addToPerClass(classVal, weight);
                addToPerClassPerBag(bagIndex, classVal, weight);
            }
        }
        addToPerBag(bagIndex, sumOfWeights);
        totaL += sumOfWeights;
    }

    /**
     * Adds given instance to all bags weighting it according to given weights.
     *
     * @param instance
     * @param weights
     * @throws Exception if something goes wrong
     */
    @Override
    public void addWeights(Instance instance, double[] weights) throws Exception {
        double classValue = instance.classValue();
        int b;

        for (b = 0; b < numBags(); b++) {
            double weight = instance.weight() * weights[b];

            addToPerBag(b, weight);
            addToPerClass(classValue, weight);
            addToPerClassPerBag(b, classValue, weight);
            totaL += weight;
        }
    }

    /**
     * Checks if at least two bags contain a minimum number of instances.
     *
     * @param minNoObj
     */
    @Override
    public boolean check(double minNoObj) {
        int count = 0;
        int b;

        for (b = 0; b < numBags(); b++) {
            if (Utils.grOrEq(perBag(b), minNoObj)) {
                count++;
            }
        }
        return count > 1;
    }

    /**
     * Clones distribution (Deep copy of distribution).
     */
    @Override
    public Object clone() throws CloneNotSupportedException {
        Object o = super.clone();

        int b;
        EJMLDistribution newDistribution = new EJMLDistribution(numBags(), numClasses());

        for (b = 0; b < numBags(); b++) {
            newDistribution.m_perBag[b] = m_perBag[b];
        }
        newDistribution.m_perClass = new Double2DoubleOpenHashMap(this.m_perClass);
        newDistribution.m_perClassPerBag = new DMatrixSparseTriplet(m_perClassPerBag);
        newDistribution.totaL = totaL;
        newDistribution.m_numClasses = m_numClasses;

        return newDistribution;
    }

    /**
     * Deletes given instance from given bag.
     *
     * @param bagIndex
     * @param instance
     * @throws Exception if something goes wrong
     */
    @Override
    public void del(int bagIndex, Instance instance) throws Exception {
        double classVal = instance.classValue();
        double weight = instance.weight();

        if (weight == 0) {
            return;
        }

        addToPerBag(bagIndex, -weight);
        addToPerClass(classVal, -weight);
        addToPerClassPerBag(bagIndex, classVal, -weight);
        totaL -= weight;
    }

    /**
     * Deletes all instances in given range from given bag.
     *
     * @param bagIndex
     * @param source
     * @param startIndex
     * @param lastPlusOne
     * @throws Exception if something goes wrong
     */
    @Override
    public void delRange(int bagIndex, Instances source, int startIndex, int lastPlusOne) throws Exception {
        double sumOfWeights = 0;
        double classVal;
        double weight;
        int i;

        for (i = startIndex; i < lastPlusOne; i++) {
            weight = source.instance(i).weight();
            if (weight != 0) {
                sumOfWeights += weight;
                classVal = source.instance(i).classValue();
                addToPerClass(classVal, -weight);
                addToPerClassPerBag(bagIndex, classVal, -weight);
            }
        }
        addToPerBag(bagIndex, -sumOfWeights);
        totaL -= sumOfWeights;
    }

    /**
     * Prints distribution.
     */
    @Override
    public String dumpDistribution() {
        StringBuffer text = new StringBuffer();
        int b, c;

        for (b = 0; b < numBags(); b++) {
            text.append("Bag num " + b + "\n");
            for (c = 0; c < numClasses(); c++) {
                text.append("Class num " + c + " " + perClassPerBag(b, c) + "\n");
            }
        }
        return text.toString();
    }

    /**
     * Sets all counts to zero.
     */
    @Override
    public void initialize() {
        for (int b = 0; b < numBags(); b++) {
            m_perBag[b] = 0;
        }
        m_perClassPerBag.zero();
        m_perClass.clear();
        totaL = 0;
    }

    /**
     * Returns matrix with distribution of class values.
     */
    @Override
    public double[][] matrix() {
        int b, c;
        double[][] perClassPerBag = new double[numBags()][numClasses()];

        for (b = 0; b < numBags(); b++) {
            for (c = 0; c < numClasses(); c++) {
                perClassPerBag[b][c] = perClassPerBag(b, c);
            }
        }
        return perClassPerBag;
    }

    /**
     * Returns index of bag containing maximum number of instances.
     */
    @Override
    public int maxBag() {
        double max;
        int maxIndex;
        int i;

        max = 0;
        maxIndex = -1;
        for (i = 0; i < numBags(); i++) {
            if (Utils.grOrEq(perBag(i), max)) {
                max = perBag(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Returns class with highest frequency over all bags.
     */
    @Override
    public int maxClass() {
        double max;
        int maxIndex;
        int i;

        max = 0;
        maxIndex = -1;
        for (i = 0; i < numClasses(); i++) {
            if (Utils.grOrEq(perClass(i), max)) {
                max = perClass(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Returns class with highest frequency for given bag.
     *
     * @param index
     */
    @Override
    public int maxClass(int index) {
        double max;
        int maxIndex;
        int i;

        max = 0;
        maxIndex = -1;
        for (i = 0; i < numClasses(); i++) {
            if (Utils.grOrEq(perClassPerBag(index, i), max)) {
                max = perClassPerBag(index, i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Returns number of bags.
     */
    @Override
    public int numBags() {
        return m_perBag.length;
    }

    /**
     * Returns number of classes.
     */
    @Override
    public int numClasses() {
        return m_numClasses;
    }

    /**
     * Returns perClass(maxClass()).
     */
    @Override
    public double numCorrect() {
        return perClass(maxClass());
    }

    /**
     * Returns perClassPerBag(index,maxClass(index)).
     *
     * @param index
     */
    @Override
    public double numCorrect(int index) {
        return perClassPerBag(index, maxClass(index));
    }

    /**
     * Returns total-numCorrect().
     */
    @Override
    public double numIncorrect() {
        return total() - numCorrect();
    }

    /**
     * Returns perBag(index)-numCorrect(index).
     *
     * @param index
     */
    @Override
    public double numIncorrect(int index) {
        return perBag(index) - numCorrect(index);
    }

    /**
     * Returns number of (possibly fractional) instances of given class in given
     * bag.
     *
     * @param bagIndex
     * @param classIndex
     */
    @Override
    public double perClassPerBag(int bagIndex, int classIndex) {
        return m_perClassPerBag.get(bagIndex, classIndex);
    }

    /**
     * Returns number of (possibly fractional) instances in given bag.
     *
     * @param bagIndex
     */
    @Override
    public double perBag(int bagIndex) {
        return m_perBag[bagIndex];
    }

    /**
     * Returns number of (possibly fractional) instances of given class.
     *
     * @param classIndex
     */
    @Override
    public double perClass(int classIndex) {
        return m_perClass.get(classIndex);
    }

    /**
     * Returns relative frequency of class over all bags with Laplace correction.
     *
     * @param classIndex
     */
    @Override
    public double laplaceProb(int classIndex) {
        return (perClass(classIndex) + 1) / (total() + numClasses());
    }

    /**
     * Returns relative frequency of class for given bag.
     *
     * @param classIndex
     * @param intIndex
     */
    @Override
    public double laplaceProb(int classIndex, int intIndex) {
        if (Utils.gr(perBag(intIndex), 0)) {
            return (perClassPerBag(intIndex, classIndex) + 1.0) / (perBag(intIndex) + numClasses());
        }
        else {
            return laplaceProb(classIndex);
        }
    }

    /**
     * Returns relative frequency of class over all bags.
     *
     * @param classIndex
     */
    @Override
    public double prob(int classIndex) {
        if (total() > 0) {
            return perClass(classIndex) / total();
        }
        return 0;
    }

    /**
     * Returns relative frequency of class for given bag.
     *
     * @param classIndex
     * @param intIndex
     */
    @Override
    public double prob(int classIndex, int intIndex) {
        if (perBag(intIndex) > 0) {
            return perClassPerBag(intIndex, classIndex) / perBag(intIndex);
        }
        return 0;
    }

    /**
     * Subtracts the given distribution from this one. The results has only one
     * bag.
     *
     * @param toSubtract
     */
    @Override
    public Distribution subtract(Distribution toSubtract) {
        EJMLDistribution newDist = new EJMLDistribution(1, numClasses());

        newDist.m_perBag[0] = total() - toSubtract.total();
        newDist.totaL = newDist.m_perBag[0];

        for (int c = 0; c < numClasses(); c++) {
            newDist.addToPerClass(c, perClass(c) - toSubtract.perClass(c));
            newDist.addToPerClassPerBag(0, c, perClass(c) - toSubtract.perClass(c));
        }
        return newDist;
    }

    /**
     * Returns total number of (possibly fractional) instances.
     */
    @Override
    public double total() {
        return totaL;
    }

    /**
     * Shifts given instance from one bag to another one.
     *
     * @param from
     * @param to
     * @param instance
     * @throws Exception if something goes wrong
     */
    @Override
    public void shift(int from, int to, Instance instance) throws Exception {
        double classVal = instance.classValue();
        double weight = instance.weight();

        addToPerClassPerBag(from, classVal, -weight);
        addToPerClassPerBag(to, classVal, weight);
        addToPerBag(from, -weight);
        addToPerBag(to, weight);
    }

    /**
     * Shifts all instances in given range from one bag to another one.
     *
     * @param from
     * @param to
     * @param source
     * @param startIndex
     * @param lastPlusOne
     * @throws Exception if something goes wrong
     */
    @Override
    public void shiftRange(int from, int to, Instances source, int startIndex, int lastPlusOne) throws Exception {
        double classVal;
        double weight;

        for (int i = startIndex; i < lastPlusOne; i++) {
            classVal = source.instance(i).classValue();
            weight = source.instance(i).weight();
            addToPerClassPerBag(from, classVal, -weight);
            addToPerClassPerBag(to, classVal, weight);
            addToPerBag(from, -weight);
            addToPerBag(to, weight);
        }
    }

    /**
     * Gets the percentage density (the percentage of values in the standard distribution that are non-zero) of the distribution
     */
    @Override
    public double getDensity() throws Exception {
        int nonZeroCount = 0;
        for (int b = 0; b < numBags(); b++) {
            for (int c = 0; c < numClasses(); c++) {
                if (perClassPerBag(b, c) != 0) {
                    nonZeroCount++;
                }
            }
        }
        return ((double) nonZeroCount) / (numBags() * numClasses()) * 100;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision$");
    }
}
