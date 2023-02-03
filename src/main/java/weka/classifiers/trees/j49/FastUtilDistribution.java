package weka.classifiers.trees.j49;

import java.util.Enumeration;

import it.unimi.dsi.fastutil.doubles.Double2DoubleMap;
import it.unimi.dsi.fastutil.doubles.Double2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.doubles.Double2ObjectOpenHashMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

public class FastUtilDistribution implements Distribution {

    //weights per bag
    Double2DoubleOpenHashMap m_perBag;

    //weights per class in Double2DoubleMap<double classVal, double weight>
    Double2DoubleOpenHashMap m_perClass;

    //weights per class per bag with a hashmap of Double2DoubleMaps
    //The key in the main map is the bag and maps to a double2double map that represents the weights per class
    Double2ObjectOpenHashMap<Double2DoubleOpenHashMap> m_perClassPerBag;

    double totaL;

    //total number of classes - must be stored since it is not retained by m_perClass or m_perClassPerBag
    int m_numClasses;

    //total number of bags - must be stored since it is not retained by m_perBag
    int m_numBags;

    /**
     * Creates and initializes a new distribution.
     */
    public FastUtilDistribution(int numBags, int numClasses) {
        m_perBag = new Double2DoubleOpenHashMap();
        m_perClass = new Double2DoubleOpenHashMap();
        m_perClassPerBag = new Double2ObjectOpenHashMap<>();
        totaL = 0;
        m_numClasses = numClasses;
        m_numBags = numBags;
    }

    /**
     * Creates and initializes a new distribution using the given array of format StandardDistribution.m_perClassPerBag
     */
    public FastUtilDistribution(double[][] table) {
        m_numClasses = table[0].length;
        m_numBags = table.length;
        m_perBag = new Double2DoubleOpenHashMap(table.length);
        m_perClass = new Double2DoubleOpenHashMap(table[0].length);
        m_perClassPerBag = new Double2ObjectOpenHashMap<>(table.length);
        totaL = 0;


        for (int b = 0; b < table.length; b++) {
            m_perClassPerBag.put(b,new Double2DoubleOpenHashMap(table[b].length));
            for (int c = 0; c < table[b].length; c++) {
                if (table[b][c] != 0) {
                    m_perBag.addTo(b,table[b][c]);
                    m_perClass.addTo(c, table[b][c]);
                    m_perClassPerBag.get(b).addTo(c, table[b][c]);

                    totaL += table[b][c];
                }
            }
            m_perClassPerBag.get(b).trim();
            if( m_perClassPerBag.get(b).isEmpty()) m_perClassPerBag.remove(b);
        }
        m_perClassPerBag.trim();
        m_perClass.trim();
        m_perBag.trim();
    }

    /**
     * Creates a distribution with only one bag according to instances in source.
     *
     * @throws Exception if something goes wrong
     */
    public FastUtilDistribution(Instances source) throws Exception {
        m_perBag = new Double2DoubleOpenHashMap(1);
        m_perClass = new Double2DoubleOpenHashMap(source.numClasses());
        m_perClassPerBag = new Double2ObjectOpenHashMap<>(1);
        m_perClassPerBag.put(0,new Double2DoubleOpenHashMap(source.numClasses()));
        totaL = 0;
        m_numClasses = source.numClasses();
        m_numBags = 1;

        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            add(0, enu.nextElement());
        }

        m_perBag.trim();
        m_perClass.trim();
        m_perClassPerBag.get(0).trim();
    }

    /**
     * Creates a distribution according to given instances and split model.
     *
     * @throws Exception if something goes wrong
     */

    public FastUtilDistribution(Instances source, ClassifierSplitModel modelToUse) throws Exception {
        int index;
        Instance instance;
        double[] weights;

        totaL = 0;
        m_numClasses = source.numClasses();
        m_numBags = modelToUse.numSubsets();

        m_perBag = new Double2DoubleOpenHashMap(modelToUse.numSubsets());
        m_perClass = new Double2DoubleOpenHashMap(m_numClasses);
        m_perClassPerBag = new Double2ObjectOpenHashMap<>(modelToUse.numSubsets());

        Enumeration<Instance> enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = enu.nextElement();
            index = modelToUse.whichSubset(instance);
            m_perClassPerBag.putIfAbsent((double) index, new Double2DoubleOpenHashMap(numClasses()));
            if (index != -1) {
                add(index, instance);
            }
            else {
                weights = modelToUse.weights(instance);
                addWeights(instance, weights);
            }
        }
        m_perBag.trim();
        m_perClass.trim();

        for (Double bag: m_perClassPerBag.keySet()) {
            m_perClassPerBag.get(bag).trim();
        }

    }

    /**
     * Creates distribution with only one bag by merging all bags of given
     * distribution.
     */
    public FastUtilDistribution(Distribution toMerge) {
        m_perBag = new Double2DoubleOpenHashMap(1);
        m_perBag.put(0,toMerge.total());

        m_perClass = new Double2DoubleOpenHashMap(toMerge.numClasses());

        m_perClassPerBag = new Double2ObjectOpenHashMap<>(1);
        m_perClassPerBag.put(0, new Double2DoubleOpenHashMap(toMerge.numClasses()));

        for(int i = 0; i<toMerge.numClasses(); i++){
            if(toMerge.perClass(i)!=0) {
                m_perClass.addTo(i, toMerge.perClass(i));
                m_perClassPerBag.get(0).addTo(i, toMerge.perClass(i));
            }
        }

        totaL = toMerge.total();
        m_numClasses = toMerge.numClasses();
        m_numBags = 1;

        m_perClass.trim();
        m_perBag.trim();
        m_perClassPerBag.get(0).trim();

    }

    /**
     * Creates distribution with two bags by merging all bags apart of the
     * indicated one.
     */
    public FastUtilDistribution(Distribution toMerge, int index) {
        int c;
        double weight;

        totaL = toMerge.total();
        m_numClasses = toMerge.numClasses();
        m_numBags = 2;

        m_perBag  = new Double2DoubleOpenHashMap(2);
        m_perBag.put(0, toMerge.perBag(index));
        m_perBag.put(1, toMerge.total() - m_perBag.get(0));

        m_perClass = new Double2DoubleOpenHashMap(numClasses());

        m_perClassPerBag = new Double2ObjectOpenHashMap<>(2);
        m_perClassPerBag.put(0, new Double2DoubleOpenHashMap(numClasses()));
        m_perClassPerBag.put(1, new Double2DoubleOpenHashMap(numClasses()));

        for(c = 0; c<toMerge.numClasses(); c++){
            weight = toMerge.perClass(c);
            if(weight!=0){
                m_perClass.put(c, weight);
            }
            if(toMerge.perClassPerBag(index, c) != 0){
                m_perClassPerBag.get(0).put(c, toMerge.perClassPerBag(index, c));
            }
            weight -= toMerge.perClassPerBag(index, c);
            if( weight != 0) {
                m_perClassPerBag.get(1).put(c,weight);
            }
        }

        m_perClass.trim();
        m_perBag.trim();
        m_perClassPerBag.get(0).trim();
        m_perClassPerBag.get(1).trim();
    }

    /**
     * Returns number of non-empty bags of distribution.
     */
    @Override
    public int actualNumBags() {
        return m_perBag.size();
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

        if(m_perClassPerBag.containsKey(bagIndex)){
            return m_perClassPerBag.get(bagIndex).size();
        }
        return 0;
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

        if(weight==0) return;

        m_perBag.addTo(bagIndex, weight);
        m_perClass.addTo(classVal, weight);

        m_perClassPerBag.putIfAbsent((double) bagIndex, new Double2DoubleOpenHashMap());
        m_perClassPerBag.get(bagIndex).addTo(classVal, weight);
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

        m_perBag.addTo(bagIndex, -weight);
        m_perClass.addTo(classVal, -weight);
        m_perClassPerBag.get(bagIndex).addTo(classVal, -weight);
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

        m_perBag.addTo(bagIndex, sum);

        for (c = 0; c < counts.length; c++) {
            if (counts[c] != 0) {
                m_perClass.addTo(c, counts[c]);
                m_perClassPerBag.get(bagIndex).addTo(c, counts[c]);
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
            if (Utils.eq(totaL, 0)) {
                probs[b] = 1.0 / probs.length;
            }
            else {
                probs[b] = perBag(b)/totaL;
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

                for (b = 0; b < numBags(); b++) {
                    newWeight = probs[b] * weight;
                    m_perClassPerBag.putIfAbsent((double) b, new Double2DoubleOpenHashMap());
                    m_perClassPerBag.get(b).addTo(classIndex, newWeight);
                    m_perBag.addTo(b, newWeight);
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

        for (b = 0; b < numBags(); b++) {
            double weight = instance.weight() * weights[b];

            if(weight!=0) {
                m_perBag.addTo(b, weight);
                m_perClass.addTo(classValue, weight);
                m_perClassPerBag.putIfAbsent((double) b, new Double2DoubleOpenHashMap());
                m_perClassPerBag.get(b).addTo(classValue,weight);
                totaL += weight;
            }
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
            if (m_perBag.containsKey(b) && Utils.grOrEq(m_perBag.get(b), minNoObj)) {
                count++;
            }
        }
        return count > 1;

    }

    /**
     * Clones distribution (Deep copy of distribution).
     */
    @Override
    public Object clone() throws CloneNotSupportedException{
        Object o = super.clone();

        int b;
        FastUtilDistribution newDistribution = new FastUtilDistribution(numBags(), numClasses());

        for (b = 0; b < numBags(); b++) {
            newDistribution.m_perBag.addTo(b, perBag(b));
            newDistribution.m_perClassPerBag.putIfAbsent((double) b, new Double2DoubleOpenHashMap(m_perClassPerBag.get(b)));
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

        m_perBag.addTo(bagIndex, -weight);
        m_perClass.addTo(classValue, -weight);
        m_perClassPerBag.get(bagIndex).addTo(classValue, -weight);
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
            m_perClassPerBag.get(bagIndex).addTo(classValue, -weight);
        }
        m_perBag.addTo(bagIndex, -sumOfWeights);
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
                text.append("Class num " + c + " " + perClassPerBag(b,c) + "\n");
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
            if( m_perClassPerBag.containsKey(b)){
                m_perClassPerBag.get(b).clear();
            }
        }
        m_perClassPerBag.clear();
        m_perBag.clear();
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
                perClassPerBag[b][c] = perClassPerBag(b,c);
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

        for (Double2DoubleMap.Entry entry : m_perClassPerBag.get(index).double2DoubleEntrySet()) {
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
        return m_numBags;
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
        if(m_perClassPerBag.containsKey(bagIndex)) {
            return m_perClassPerBag.get(bagIndex).get(classIndex);
        }
        return 0;
    }

    /**
     * Returns number of (possibly fractional) instances in given bag.
     *
     * @param bagIndex
     */
    @Override
    public double perBag(int bagIndex) {
        return m_perBag.get(bagIndex);
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
        return (m_perClass.get(classIndex) + 1) / (total() + numClasses());
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

        FastUtilDistribution newDist = new FastUtilDistribution(1, numClasses());

        newDist.m_perBag.put(0, total() - toSubtract.total());
        newDist.totaL = newDist.perBag(0);

        for (int c = 0; c < numClasses(); c++) {
            newDist.m_perClassPerBag.putIfAbsent((double) 0, new Double2DoubleOpenHashMap());
            newDist.m_perClassPerBag.get(0).put(c,perClass(c) - toSubtract.perClass(c));
            newDist.m_perClass.put(c, newDist.perClassPerBag(0,c));
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

        m_perClassPerBag.get(from).addTo(classValue, -weight);
        m_perClassPerBag.get(to).addTo(classValue, -weight);
        m_perBag.addTo(from, -weight);
        m_perBag.addTo(to, weight);
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
            m_perClassPerBag.get(from).addTo(classValue, -weight);
            m_perClassPerBag.putIfAbsent( (double) to, new Double2DoubleOpenHashMap());
            m_perClassPerBag.get(to).addTo(classValue,weight);
            m_perBag.addTo(from, -weight);
            m_perBag.addTo(to, weight);
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
