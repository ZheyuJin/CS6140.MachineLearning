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

  private double acc(List<Boolean> lst) {
    int trueCount = 0;
    for (boolean b : lst)
      if (b == true)
        trueCount++;

    return (double) trueCount / lst.size();
  }

  private double avg(List<Double> lst) {
    double sum = 0;
    for (double d : lst)
      sum += d;

    return sum / lst.size();
  }

  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

  void confuseMatrixDataCalc(double label, double predict) {
    if (predict == 1) {
      if (label == predict)
        tp++;
      else
        fp++;
    } else {
      if (label == predict)
        tn++;
      else
        fn++;
    }
  }

  void exec() {
    // List<Double> trainingMSEList = new ArrayList<Double>();
    // List<Double> testMSEList = new ArrayList<Double>();
    List<Boolean> trainACCList = new ArrayList<>();
    List<Boolean> testACCList = new ArrayList<>();


    for (int i = 0; i < kfold; i++) {
      System.out.println(i);
      init(); // load data.

      assert null != trainMatrix;
      assert null != testMatrix;

      tree = new ClassificationTree(trainMatrix, testMatrix, labelIdx, stopRatio);
      tree.train();

      /* get train ACC */
      for (Node n : tree.leafList) {
        for (int row : n.tdpIdxList) {
          trainACCList.add(n.lable == trainMatrix[row][labelIdx]);
          confuseMatrixDataCalc(n.lable, trainMatrix[row][labelIdx]);
        }
      }
      // trainMSE /= trainMatrix.length;
      // trainingMSEList.add(trainMSE);

      /* get test ACC */
      for (double[] dp : testMatrix) {
        double outLabel = tree.test(dp);
        double actualLabel = dp[labelIdx];
        testACCList.add(outLabel == actualLabel);
        confuseMatrixDataCalc(actualLabel, outLabel);
      }
    }

    double trainACC = acc(trainACCList);
    double testACC = acc(testACCList);

    System.out.printf("stopRatio %.3f \t tranACC avg %.3f \t testACC avg %.3f\n", stopRatio,
        trainACC, testACC);

    double totalCount = (double)(tp + fp + tn + fn);
    System.out.printf("truePos %.4f \t falsePos %.4f \t trueNeg %.4f \t falseNeg %.4f \t", tp
        / totalCount, fp / totalCount, tn / totalCount, fn / totalCount);
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
