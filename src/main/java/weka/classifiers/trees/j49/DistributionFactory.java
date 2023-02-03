package weka.classifiers.trees.j49;

import java.io.Serializable;
import weka.core.Instances;



public class DistributionFactory implements Serializable {

    //To change the type of distribution used by J49, change this value.
    //0 - Standard; 1 - Array of Maps; 2 - Maps of Maps; 3 - EJML
    int distributionType = 1;

    public DistributionFactory(int type){
        distributionType = type;
    }

    public DistributionFactory(){};

    //The following methods initialize m_distribution with the correct type
    //The different versions correspond to the different Distribution constructors that are present in split classes

    /** returns new Distribution(numBags, numClasses) */
    public Distribution getDistribution(int numBags, int numClasses){
        //if( numBags > 10 ) return new StandardDistribution(numBags, numClasses);
        switch(distributionType){
            case 1:
                return new ArrayAndFastUtilDistribution(numBags, numClasses);
            case 2:
                return new FastUtilDistribution(numBags, numClasses);
            case 3:
                return new EJMLDistribution(numBags, numClasses);
            default:
                return new OriginalDistribution(numBags, numClasses);
        }
    }

    /** returns new Distribution(double[][] table) */
    public Distribution getDistribution(double[][] table){

        /*StandardDistribution testDist = new StandardDistribution(table);
        if (testDist.perClassPerBagDensity() > 5) {
            return testDist;
        }*/

        switch(distributionType){
            case 1:
                return new ArrayAndFastUtilDistribution(table);
            case 2:
                return new FastUtilDistribution(table);
            case 3:
                return new EJMLDistribution(table);
            default:
                return new OriginalDistribution(table);
        }
    }

    /** returns new Distribution(source, model) */
    public Distribution getDistribution(Instances source) throws Exception {
        /*StandardDistribution testDist = new StandardDistribution(source);
        if( testDist.perClassPerBagDensity() > 5 ){
            return testDist;
        }*/
        switch(distributionType){
            default:
                return new OriginalDistribution(source);
            case 1:
                return new ArrayAndFastUtilDistribution(source);
            case 2:
                return new FastUtilDistribution(source);
            case 3:
                return new EJMLDistribution(source);
        }
    }

    /** returns new Distribution(source, model) */
    public Distribution getDistribution(Instances source, ClassifierSplitModel model) throws Exception {
        /*StandardDistribution testDist = new StandardDistribution(source, model);
        if( testDist.perClassPerBagDensity() > 5 ){
            return testDist;
        }*/

        switch(distributionType){
            default:
                return new OriginalDistribution(source, model);
            case 1:
                return new ArrayAndFastUtilDistribution(source, model);
            case 2:
                return new FastUtilDistribution(source,model);
            case 3:
                return new EJMLDistribution(source, model);
        }
    }

    /** returns new Distribution(toMerge, index) */
    public Distribution getDistribution(Distribution toMerge, int index){

        /*StandardDistribution testDist = new StandardDistribution(toMerge, index);
        if( testDist.perClassPerBagDensity() > 5 ){
            return testDist;
        }*/

        switch(distributionType){
            default:
                return new OriginalDistribution(toMerge, index);
            case 1:
                return new ArrayAndFastUtilDistribution(toMerge, index);
            case 2:
                return new FastUtilDistribution(toMerge, index);
            case 3:
                return new EJMLDistribution(toMerge, index);
        }
    }

    /** returns new Distribution(toMerge) */
    public Distribution getDistribution(Distribution toMerge){

        /*StandardDistribution testDist = new StandardDistribution(toMerge);
        if( testDist.perClassPerBagDensity() > 5 ){
            return testDist;
        }*/

        switch(distributionType){
            default:
                return new OriginalDistribution(toMerge);
            case 1:
                return new ArrayAndFastUtilDistribution(toMerge);
            case 2:
                return new FastUtilDistribution(toMerge);
            case 3:
                return new EJMLDistribution(toMerge);
        }
    }

}
