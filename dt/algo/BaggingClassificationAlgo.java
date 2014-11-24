package dt.algo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import dt.dataloader.IMatrixLoader;
import dt.dataloader.SamplingSpamLoader;
import dt.dataloader.SpamMatrixLoader;


public class BaggingClassificationAlgo extends Algo {
  static final int labelIndex = 57;
  /**
   * stop if a node size is smaller than this ratio
   */
  static final double stopRatio = 0.15;
  static final int kfold = 10;

  public BaggingClassificationAlgo(IMatrixLoader ld) {
    super(ld, labelIndex);
  }


  private double avg(List<Double> lst) {
    double sum = 0;
    for (double d : lst)
      sum += d;

    return sum / lst.size();
  }

  void exec() {
    throw new UnsupportedOperationException();
  }

  /*
   * double traningMSE = 0; double testMSE = 0;
   */
  double[] predict() {

    System.out.println();
    init(); // load data.

    assert null != trainMatrix;
    assert null != testMatrix;

    tree = new ClassificationTree(trainMatrix, testMatrix, labelIdx, stopRatio);
    tree.train();

    double[] predictino = new double[testMatrix.length];

    for (int i = 0; i < testMatrix.length; i++) {
      predictino[i] = tree.test(testMatrix[i]);
    }
    return predictino;
  }


  class ClassificationTree extends DTree {

    ClassificationTree(double[][] train, double[][] test, int labelIdx, double ratio) {
      super(train, test, labelIdx, ratio);
    }


    final class FeatureComp implements Comparator<Integer> {
      int fIndex; // feature index

      public FeatureComp(int featureIdx) {
        fIndex = featureIdx;
      }

      @Override
      public int compare(Integer here, Integer there) {
        double comp = trainingData[here][fIndex] - trainingData[there][fIndex];
        /* Very bad idea to compare double with int!!. */
        if (comp > 0)
          return 1;
        else if (comp < 0)
          return -1;
        else
          return 0;
      }

    }

    @Override
    /**
     * choose a feature and index so that these maximizes IG
     * */
    Pair<Integer, Double> bestSplit(List<Integer> dpIdxList) {
      final double parantEntropy = calcEntropy(dpIdxList);

      double bestGain = 0;
      int bestFeatIndex = 0, bestFeatRow = 0, totalSize = dpIdxList.size();
      LinkedList<Integer> left = new LinkedList<>(), right = new LinkedList<>();

      for (int j = 0; j < labeIdx - 1; j++) { // label is last feature.
        Collections.sort(dpIdxList, new FeatureComp(j));

        left.clear();
        right.clear();

        right.addAll(dpIdxList);

        while (right.size() > 1) {
          left.add(right.removeFirst()); // giving one to left.

          if (left.size() != 1) {
            while (left.getLast() == right.getFirst())
              left.add(right.removeFirst()); // giving one to left.
          }

          // conditional information entropy;
          double childrenEntropy =
              (calcEntropy(left) * left.size() + calcEntropy(right) * right.size()) / totalSize;
          double gain = parantEntropy - childrenEntropy;

          if (gain > bestGain) {
            bestGain = gain;
            bestFeatIndex = j;
            bestFeatRow = left.size() - 1;
          }
        }
      }

      return new Pair<Integer, Double>(bestFeatIndex, trainingData[bestFeatRow][bestFeatIndex]);
    }

    /**
     * calculate entropy
     */
    private double calcEntropy(List<Integer> lst) {
      double hy = 0;

      int[] map = new int[2]; // only two label.

      for (int idx : lst) {
        double label = trainingData[idx][labelIdx];
        map[(int) label]++;
      }


      int size = map[0] + map[1];
      for (int count : map) {
        if (count == 0)
          continue;
        double p = (double) count / size;
        hy += (p * Math.log(1 / p));
      }

      double ret = hy / Math.log(2);
      return ret;
    }

    @Override
    double decideLabel(List<Integer> dpIdxList) {
      int[] count = new int[2];

      for (int i : dpIdxList) {
        // very bad idea. do not use unless label is natural number contained in double type.
        int label = (int) trainingData[i][labeIdx];
        count[label]++;
      }

      if (count[0] > count[1]) // more 0
        return 0;
      else
        return 1;
    }

  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub
    int T = 1;
    double[][] totalLabel = new double[T][];

    for (int i = 0; i < T; i++) {
      System.out.println("iteration " +i);
      IMatrixLoader ld = new SamplingSpamLoader();
      BaggingClassificationAlgo algo = new BaggingClassificationAlgo(ld);
      totalLabel[i] = algo.predict();
    }

    // vote
    double[] outlabel = vote(totalLabel);

    // check acc
    SpamMatrixLoader ld = new SpamMatrixLoader();
    double[][] testmatrix = ld.loadTestMatrix();
    double[] testlabel = new double[testmatrix.length];

    int idx = 0;
    for (double[] line : testmatrix)
      testlabel[idx++] = line[line.length - 1];

    System.out.println("testlabel len " + testlabel.length);
    System.out.println("outlabel len " + outlabel.length);
    int accCount = 0;
    for (int i = 0; i < outlabel.length; i++)
      if (testlabel[i] == outlabel[i]) // float comparison..
        accCount++;

    System.out.printf("bagging acc= %f", accCount / testlabel.length);
  }

  private static double sum(double[] array){
    double ans =0;
    for(double d: array)
      ans += d;
    return ans;
  }
  
  private static double[] vote(double[][] totalLabel) {
    double[] ans = new double[totalLabel[0].length];
    int labelcount = totalLabel.length;
    
    for(double[] array: totalLabel){
      for(int i=0; i< array.length; i++){
        ans[i] += array[i];
      }
    }
    
    for(int i=0; i< ans.length; i++)
      ans[i]= Math.round(ans[i]/labelcount);
    
    return ans;
  }
}
