package weka.classifiers.trees.j49;

import java.util.Enumeration;

import it.unimi.dsi.fastutil.doubles.Double2DoubleMap;
import it.unimi.dsi.fastutil.doubles.Double2DoubleOpenHashMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

public class ArrayAndFastUtilDistribution implements Distribution {

    //weights per bag
    double[] m_perBag;

    //weights per class in Double2DoubleMap<double classVal, double weight>
    Double2DoubleOpenHashMap m_perClass;

    //weights per class per bag, with an array of Double2DoubleMaps<double classValue, double weight>,
    //Array index represents bag value
    Double2DoubleOpenHashMap[] m_perClassPerBag;

    double totaL;

    //total number of classes - must be stored since it is not retained by m_perClass or m_perClassPerBag
    int m_numClasses;

    /**
     * Creates and initializes a new distribution.
     */
    public ArrayAndFastUtilDistribution(int numBags, int numClasses) {
        m_perBag = new double[numBags];
        m_perClass = new Double2DoubleOpenHashMap();
        m_perClassPerBag = new Double2DoubleOpenHashMap[numBags];
        for (int i = 0; i < numBags; i++) {
            m_perClassPerBag[i] = new Double2DoubleOpenHashMap();
        }
        totaL = 0;
        m_numClasses = numClasses;
    }

    /**
     * Creates and initializes a new distribution using the given array of format StandardDistribution.m_perClassPerBag
     */
    public ArrayAndFastUtilDistribution(double[][] table) {
        m_perBag = new double[table.length];
        m_perClass = new Double2DoubleOpenHashMap(table[0].length);
        m_perClassPerBag = new Double2DoubleOpenHashMap[table.length];
        totaL = 0;
        m_numClasses = table[0].length;

        for (int i = 0; i < table.length; i++) {
            m_perClassPerBag[i] = new Double2DoubleOpenHashMap(table[0].length);
        }

        for (int b = 0; b < table.length; b++) {
            for (int c = 0; c < table[b].length; c++) {
                if (table[b][c] != 0) {
                    m_perBag[b] += table[b][c];
                    m_perClass.addTo(c, table[b][c]);
                    m_perClassPerBag[b].addTo(c, table[b][c]);
                    totaL += table[b][c];
                }
            }
            m_perClassPerBag[b].trim();
        }
        m_perClass.trim();
    }

    /**
     * Creates a distribution with only one bag according to instances in source.
     *
     * @throws Exception if something goes wrong
     */
    public ArrayAndFastUtilDistribution(Instances source) throws Exception {
        m_perBag = new double[1];
        m_perClass = new Double2DoubleOpenHashMap(source.numClasses());
        m_perClassPerBag = new Double2DoubleOpenHashMap[1];
        m_perClassPerBag[0] = new Double2DoubleOpenHashMap();
        totaL = 0;
        m_numClasses = source.numClasses();

        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            add(0, enu.nextElement());
        }

        m_perClass.trim();
        m_perClassPerBag[0].trim();
    }

    /**
     * Creates a distribution according to given instances and split model.
     *
     * @throws Exception if something goes wrong
     */

    public ArrayAndFastUtilDistribution(Instances source, ClassifierSplitModel modelToUse) throws Exception {
        int index;
        Instance instance;
        double[] weights;

        totaL = 0;
        m_numClasses = source.numClasses();

        m_perBag = new double[modelToUse.numSubsets()];
        m_perClass = new Double2DoubleOpenHashMap(m_numClasses);
        m_perClassPerBag = new Double2DoubleOpenHashMap[modelToUse.numSubsets()];
        for (int i = 0; i < m_perClassPerBag.length; i++) {
            m_perClassPerBag[i] = new Double2DoubleOpenHashMap(m_numClasses);
        }

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
        for (int i = 0; i < m_perClassPerBag.length; i++) {
            m_perClassPerBag[i].trim();
        }

    }

    /**
     * Creates distribution with only one bag by merging all bags of given
     * distribution.
     */
    public ArrayAndFastUtilDistribution(Distribution toMerge) {
        m_perBag = new double[1];
        m_perBag[0] = toMerge.total();

        m_perClass = new Double2DoubleOpenHashMap(toMerge.numClasses());
        m_perClassPerBag = new Double2DoubleOpenHashMap[1];
        m_perClassPerBag[0] = new Double2DoubleOpenHashMap(toMerge.numClasses());

        for (int i = 0; i < toMerge.numClasses(); i++) {
            if (toMerge.perClass(i) != 0) {
                m_perClass.addTo(i, toMerge.perClass(i));
                m_perClassPerBag[0].addTo(i, toMerge.perClass(i));
            }
        }

        totaL = toMerge.total();
        m_numClasses = toMerge.numClasses();

        m_perClass.trim();
        m_perClassPerBag[0].trim();

    }

    /**
     * Creates distribution with two bags by merging all bags apart of the
     * indicated one.
     */
    public ArrayAndFastUtilDistribution(Distribution toMerge, int index) {
        int c;
        double weight;

        totaL = toMerge.total();
        m_numClasses = toMerge.numClasses();

        m_perBag = new double[2];
        m_perBag[0] = toMerge.perBag(index);
        m_perBag[1] = toMerge.total() - m_perBag[0];

        m_perClass = new Double2DoubleOpenHashMap(m_numClasses);
        m_perClassPerBag = new Double2DoubleOpenHashMap[2];
        m_perClassPerBag[0] = new Double2DoubleOpenHashMap(m_numClasses);
        m_perClassPerBag[1] = new Double2DoubleOpenHashMap(m_numClasses);

        for (c = 0; c < toMerge.numClasses(); c++) {
            m_perClass.put(c, toMerge.perClass(c));
            m_perClassPerBag[0].put(c, toMerge.perClassPerBag(index, c));
            weight = toMerge.perClass(c) - toMerge.perClassPerBag(index, c);
            if (weight != 0) {
                m_perClassPerBag[1].put(c, weight);
            }
        }

        m_perClass.trim();
        m_perClassPerBag[0].trim();
        m_perClassPerBag[1].trim();
    }

    /**
     * Returns number of non-empty bags of distribution.
     */
    @Override
    public int actualNumBags() {
        int returnValue = 0;
        int i;

        for (i = 0; i < m_perBag.length; i++) {
            if (Utils.gr(m_perBag[i], 0)) {
                returnValue++;
            }
        }

        return returnValue;
    }

    /**
     * Returns number of classes actually occuring in distribution.
     */
    @Override
    public int actualNumClasses() {

        //use map.size() method assuming there will be no zero values added
        return m_perClass.size();
    }

    /**
     * Returns number of classes actually occuring in given bag.
     *
     * @param bagIndex
     */
    @Override
    public int actualNumClasses(int bagIndex) {

        //assume there are no zero values added
        return m_perClassPerBag[bagIndex].size();
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

        m_perBag[bagIndex] += weight;
        m_perClass.addTo(classVal, weight);
        m_perClassPerBag[bagIndex].addTo(classVal, weight);
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

        double classVal = instance.classValue();
        double weight = instance.weight();

        m_perBag[bagIndex] -= weight;
        m_perClass.addTo(classVal, -weight);
        m_perClassPerBag[bagIndex].addTo(classVal, -weight);
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

        m_perBag[bagIndex] += sum;

        for (c = 0; c < counts.length; c++) {
            if (counts[c] != 0) {
                m_perClass.addTo(c, counts[c]);
                m_perClassPerBag[bagIndex].addTo(c, counts[c]);
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

        probs = new double[m_perBag.length];
        for (b = 0; b < m_perBag.length; b++) {
            if (Utils.eq(totaL, 0)) {
                probs[b] = 1.0 / probs.length;
            }
            else {
                probs[b] = m_perBag[b] / totaL;
            }
        }
        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = enu.nextElement();
            if (instance.isMissing(attIndex)) {
                classIndex = instance.classValue();
                weight = instance.weight();
                m_perClass.addTo(classIndex, weight);
                totaL += weight;

                for (b = 0; b < m_perBag.length; b++) {
                    newWeight = probs[b] * weight;
                    m_perClassPerBag[b].addTo(classIndex, newWeight);
                    m_perBag[b] += newWeight;
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

        for (int i = startIndex; i < lastPlusOne; i++) {
            add(bagIndex, source.instance(i));
        }
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

        for (b = 0; b < m_perBag.length; b++) {
            double weight = instance.weight() * weights[b];

            m_perBag[b] += weight;
            m_perClass.addTo(classValue, weight);
            m_perClassPerBag[b].addTo(classValue, weight);
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

        for (b = 0; b < m_perBag.length; b++) {
            if (Utils.grOrEq(m_perBag[b], minNoObj)) {
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
        ArrayAndFastUtilDistribution newDistribution = new ArrayAndFastUtilDistribution(numBags(), numClasses());

        for (b = 0; b < m_perBag.length; b++) {
            newDistribution.m_perBag[b] = m_perBag[b];
            newDistribution.m_perClassPerBag[b] = new Double2DoubleOpenHashMap(m_perClassPerBag[b]);
        }
        newDistribution.m_perClass = new Double2DoubleOpenHashMap(this.m_perClass);
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

        double classValue = instance.classValue();
        double weight = instance.weight();

        m_perBag[bagIndex] -= weight;
        m_perClass.addTo(classValue, -weight);
        m_perClassPerBag[bagIndex].addTo(classValue, -weight);
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
        double classValue;
        double weight;
        int i;

        for (i = startIndex; i < lastPlusOne; i++) {
            weight = source.instance(i).weight();
            sumOfWeights += weight;
            classValue = source.instance(i).classValue();
            m_perClass.addTo(classValue, -weight);
            m_perClassPerBag[bagIndex].addTo(classValue, -weight);
        }
        m_perBag[bagIndex] -= sumOfWeights;
        totaL -= sumOfWeights;
    }

    /**
     * Prints distribution.
     */
    @Override
    public String dumpDistribution() {
        StringBuffer text = new StringBuffer();
        int b, c;

        for (b = 0; b < m_perBag.length; b++) {
            text.append("Bag num " + b + "\n");
            for (c = 0; c < m_numClasses; c++) {
                text.append("Class num " + c + " " + m_perClassPerBag[b].get(c) + "\n");
            }
        }
        return text.toString();
    }

    /**
     * Sets all counts to zero.
     */
    @Override
    public void initialize() {
        for (int b = 0; b < m_perBag.length; b++) {
            m_perBag[b] = 0;
            m_perClassPerBag[b].clear();
        }
        m_perClass.clear();
        totaL = 0;

    }

    /**
     * Returns matrix with distribution of class values.
     */
    @Override
    public double[][] matrix() {
        int b, c;
        double[][] perClassPerBag = new double[m_perBag.length][m_numClasses];

        for (b = 0; b < m_perBag.length; b++) {
            for (c = 0; c < m_numClasses; c++) {
                perClassPerBag[b][c] = m_perClassPerBag[b].get(c);
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
        for (i = 0; i < m_perBag.length; i++) {
            if (Utils.grOrEq(m_perBag[i], max)) {
                max = m_perBag[i];
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
        Double2DoubleMap.Entry maxEntry = null;

        for (Double2DoubleMap.Entry entry : m_perClass.double2DoubleEntrySet()) {
            if (maxEntry == null || (entry.getDoubleValue() > maxEntry.getDoubleValue())) {
                maxEntry = entry;
            }
        }
        if (maxEntry == null) {
            return -1;
        }
        return (int) maxEntry.getDoubleKey();
    }

    /**
     * Returns class with highest frequency for given bag.
     *
     * @param index
     */
    @Override
    public int maxClass(int index) {
        Double2DoubleMap.Entry maxEntry = null;

        for (Double2DoubleMap.Entry entry : m_perClassPerBag[index].double2DoubleEntrySet()) {
            if (maxEntry == null || (entry.getDoubleValue() > maxEntry.getDoubleValue())) {
                maxEntry = entry;
            }
        }
        if (maxEntry == null) {
            return -1;
        }
        return (int) maxEntry.getDoubleKey();
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
        return totaL - numCorrect();
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
        return m_perClassPerBag[bagIndex].get(classIndex);
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
        return (m_perClass.get(classIndex) + 1) / (totaL + m_numClasses);
    }

    /**
     * Returns relative frequency of class for given bag.
     *
     * @param classIndex
     * @param intIndex
     */
    @Override
    public double laplaceProb(int classIndex, int intIndex) {

        if (Utils.gr(m_perBag[intIndex], 0)) {
            return (m_perClassPerBag[intIndex].get(classIndex) + 1.0) / (m_perBag[intIndex] + m_numClasses);
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
        if (total() > 0 ) {
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
        if (perBag(intIndex) > 0 ) {
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

        ArrayAndFastUtilDistribution newDist = new ArrayAndFastUtilDistribution(1, m_numClasses);

        newDist.m_perBag[0] = totaL - toSubtract.total();
        newDist.totaL = newDist.m_perBag[0];

        for (int c = 0; c < m_numClasses; c++) {
            newDist.m_perClassPerBag[0].put(c, perClass(c) - toSubtract.perClass(c));
            newDist.m_perClass.put(c, newDist.m_perClassPerBag[0].get(c));
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
        double classValue = instance.classValue();
        double weight = instance.weight();

        m_perClassPerBag[from].addTo(classValue, -weight);
        m_perClassPerBag[to].addTo(classValue, weight);
        m_perBag[from] -= weight;
        m_perBag[to] += weight;
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
        double classValue;
        double weight;

        for (int i = startIndex; i < lastPlusOne; i++) {
            classValue = source.instance(i).classValue();
            weight = source.instance(i).weight();
            m_perClassPerBag[from].addTo(classValue, -weight);
            m_perClassPerBag[to].addTo(classValue, weight);
            m_perBag[from] -= weight;
            m_perBag[to] += weight;
        }
    }

    /**
     * Gets the percentage density (the percentage of values in the standard distribution that are non-zero) of the distribution
     * @return the density of the distribution
     */
    @Override
    public double getDensity() throws Exception {
        int nonZeroCount = 0;
        for(int b = 0; b<numBags(); b++){
            for(int c = 0; c<numClasses(); c++){
                if(perClassPerBag(b,c)!=0){
                    nonZeroCount++;
                }
            }
        }
        return ((double) nonZeroCount)/(numBags() * numClasses()) * 100;
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
