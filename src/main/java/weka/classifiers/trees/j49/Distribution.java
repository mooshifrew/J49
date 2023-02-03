package weka.classifiers.trees.j49;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;

public interface Distribution extends Cloneable, Serializable, RevisionHandler {

    /**
     * Returns number of non-empty bags of distribution.
     */
    int actualNumBags();

    /**
     * Returns number of classes actually occuring in distribution.
     */
    int actualNumClasses();

    /**
     * Returns number of classes actually occuring in given bag.
     */
    int actualNumClasses(int bagIndex);

    /**
     * Adds given instance to given bag.
     *
     * @throws Exception if something goes wrong
     */
    void add(int bagIndex, Instance instance) throws Exception;

    /**
     * Subtracts given instance from given bag.
     *
     * @throws Exception if something goes wrong
     */
    void sub(int bagIndex, Instance instance) throws Exception;

    /**
     * Adds counts to given bag.
     */
    void add(int bagIndex, double[] counts);

    /**
     * Adds all instances with unknown values for given attribute, weighted
     * according to frequency of instances in each bag.
     *
     * @throws Exception if something goes wrong
     */
    void addInstWithUnknown(Instances source, int attIndex)
            throws Exception;

    /**
     * Adds all instances in given range to given bag.
     *
     * @throws Exception if something goes wrong
     */
    void addRange(int bagIndex, Instances source, int startIndex,
            int lastPlusOne) throws Exception;

    /**
     * Adds given instance to all bags weighting it according to given weights.
     *
     * @throws Exception if something goes wrong
     */
    void addWeights(Instance instance, double[] weights)
            throws Exception;

    /**
     * Checks if at least two bags contain a minimum number of instances.
     */
    boolean check(double minNoObj);

    /**
     * Clones distribution (Deep copy of distribution).
     */
    Object clone() throws CloneNotSupportedException;

    /**
     * Deletes given instance from given bag.
     *
     * @throws Exception if something goes wrong
     */
    void del(int bagIndex, Instance instance) throws Exception;

    /**
     * Deletes all instances in given range from given bag.
     *
     * @throws Exception if something goes wrong
     */
    void delRange(int bagIndex, Instances source, int startIndex,
            int lastPlusOne) throws Exception;

    /**
     * Prints distribution.
     */
    String dumpDistribution();

    /**
     * Sets all counts to zero.
     */
    void initialize();

    /**
     * Returns matrix with distribution of class values.
     */
    double[][] matrix();

    /**
     * Returns index of bag containing maximum number of instances.
     */
    int maxBag();

    /**
     * Returns class with highest frequency over all bags.
     */
    int maxClass();

    /**
     * Returns class with highest frequency for given bag.
     */
    int maxClass(int index);

    /**
     * Returns number of bags.
     */
    int numBags();

    /**
     * Returns number of classes.
     */
    int numClasses();

    /**
     * Returns perClass(maxClass()).
     */
    double numCorrect();

    /**
     * Returns perClassPerBag(index,maxClass(index)).
     */
    double numCorrect(int index);

    /**
     * Returns total-numCorrect().
     */
    double numIncorrect();

    /**
     * Returns perBag(index)-numCorrect(index).
     */
    double numIncorrect(int index);

    /**
     * Returns number of (possibly fractional) instances of given class in given
     * bag.
     */
    double perClassPerBag(int bagIndex, int classIndex);

    /**
     * Returns number of (possibly fractional) instances in given bag.
     */
    double perBag(int bagIndex);

    /**
     * Returns number of (possibly fractional) instances of given class.
     */
    double perClass(int classIndex);

    /**
     * Returns relative frequency of class over all bags with Laplace correction.
     */
    double laplaceProb(int classIndex);

    /**
     * Returns relative frequency of class for given bag.
     */
    double laplaceProb(int classIndex, int intIndex);

    /**
     * Returns relative frequency of class over all bags.
     */
    double prob(int classIndex);

    /**
     * Returns relative frequency of class for given bag.
     */
    double prob(int classIndex, int intIndex);

    /**
     * Subtracts the given distribution from this one. The results has only one
     * bag.
     */
    Distribution subtract(Distribution toSubtract);

    /**
     * Returns total number of (possibly fractional) instances.
     */
    double total();

    /**
     * Shifts given instance from one bag to another one.
     *
     * @throws Exception if something goes wrong
     */
    void shift(int from, int to, Instance instance) throws Exception;

    /**
     * Shifts all instances in given range from one bag to another one.
     *
     * @throws Exception if something goes wrong
     */
    void shiftRange(int from, int to, Instances source,
            int startIndex, int lastPlusOne) throws Exception;

    /**
     * Gets the percentage density (the percentage of values in the standard distribution that are non-zero) of the distribution
     */
    double getDensity() throws Exception;

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    String getRevision();
}
