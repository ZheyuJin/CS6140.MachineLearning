package dt.algo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import dt.dataloader.IMatrixLoader;
import dt.dataloader.SpamMatrixLoader;


public class ClassificationAlgo extends Algo {
  static final int labelIndex = 57;
  /**
   * stop if a node size is smaller than this ratio
   */
  static final double stopRatio = 0.15;
  static final int kfold = 10;

  public ClassificationAlgo(IMatrixLoader ld) {
    super(ld, labelIndex);
  }


  private double avg(List<Double> lst) {
    double sum = 0;
    for (double d : lst)
      sum += d;

    return sum / lst.size();
  }

  /*
   * double traningMSE = 0; double testMSE = 0;
   */
  @Override
  void exec() {
    // List<Double> trainingMSEList = new ArrayList<Double>();
    // List<Double> testMSEList = new ArrayList<Double>();
    List<Boolean> trainACC = new ArrayList<>();
    List<Boolean> testACC = new ArrayList<>();
    

    for (int i = 0; i < kfold; i++) {
      System.out.println();
      init(); // load data.

      assert null != trainMatrix;
      assert null != testMatrix;

      tree = new ClassificationTree(trainMatrix, testMatrix, labelIdx, stopRatio);
      tree.train();

      /* get train MSE */
//      double trainMSE = 0;
      for (Node n : tree.leafList) {
        for (int row : n.tdpIdxList) {
          trainMSE += Math.pow(n.lable - trainMatrix[row][labelIdx], 2);
        }
      }
      trainMSE /= trainMatrix.length;
      trainingMSEList.add(trainMSE);

      /* get test MSE */
      double testMSE = 0;
      for (double[] dp : testMatrix) {
        double outLabel = tree.test(dp);
        double actualLabel = dp[labelIdx];
        testMSE += Math.pow(outLabel - actualLabel, 2);
      }
      testMSE /= testMatrix.length;
      testMSEList.add(testMSE);

      System.out.printf(
          "\t stopRatio %.3f \t nodeCount %d \t leafCount %d \t trainMSE %.3f \t testMSE %.3f \n",
          stopRatio, tree.nodeCount, tree.leafCount, trainMSE, testMSE);
    }

    double trainError = avg(trainingMSEList);
    double testError = avg(testMSEList);

    System.out.printf("stopRatio %.3f \t tranErrorMSE avg %.3f \t testErrorMSE avg %.3f\n",
        stopRatio, trainError, testError);

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
    SpamMatrixLoader ld = new SpamMatrixLoader();
    ClassificationAlgo algo = new ClassificationAlgo(ld);
    algo.exec();
  }

}
