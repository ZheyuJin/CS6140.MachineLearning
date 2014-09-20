package dt.algo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;

import dt.dataloader.HousingMatrixLoader;
import dt.dataloader.IMatrixLoader;



public class RegressionTreeAlgo extends Algo {

  final double stopRatio = 0.20;

  /*
   * double traningMSE = 0; double testMSE = 0;
   */
  @Override
  void exec() {
    List<Double> trainingErrMSE = new ArrayList<Double>();
    List<Double> testErrMSE = new ArrayList<Double>();

    init(); // load data.
    tree = new RegressionTree(trainMatrix, testMatrix, labelIdx, stopRatio);
    tree.train();


    /* test and save result. */

    double traningMSE = calcTraningMSE();
    double testMSE = calcTestMSE();

    String report =
        String.format(
            "ratio: %.2f \t node count: %8d \t leaf count: %8d \t traningMSE: %8f \t testMSE %8f ",
            stopRatio, tree.nodeCount, tree.leafCount, traningMSE, testMSE);

    System.out.println(report);
  }

  private double calcTraningMSE() {
    double squareSum = 0;
    int size = 0;

    for (Node n : tree.leafList) {
      double outLabel = n.lable;
      for (int idx : n.tdpIdxList) {
        double actualLabel = tree.trainingData[idx][labelIdx];
        squareSum += Math.pow(actualLabel - outLabel, 2);
        size++;
      }
    }

    assert (size == tree.trainingData.length);

    return squareSum / size;
  }

  private double calcTestMSE() {
    double squareSum = 0;

    for (double[] dp : tree.testData) {
      double out = tree.test(dp);
      double actualLabel = dp[labelIdx];
      squareSum += Math.pow(out - actualLabel, 2);
    }
    System.out.printf("test squaresum %f \t size %d \n", squareSum, tree.testData.length);
    return squareSum / tree.testData.length;
  }


  class RegressionTree extends DTree {

    public RegressionTree(double[][] train, double[][] test, int labelIdx, double ratio) {
      super(train, test, labelIdx, ratio);
    }


    @Override
    /**
     * 
     * The most complex function in this project.
     *  search in d*m space the best feature and threashold value that maximizes
     *  decrease in variance.
     *  
     *  using a translated forlula to compute the decrease.
     */
    Pair<Integer, Double> bestSplit(List<Integer> dpIdxList) {
      int bestFeatIndex = 0;
      int bestThreashRow = -1;
      double maxDelta = 0;

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

      for (int j = 0; j < labeIdx - 1; j++) { // label is last feature.
        Collections.sort(dpIdxList, new FeatureComp(j));

        /*
         * for(int i: dpIdxList){ System.out.println(trainingData[i][j]); } System.out.println();
         */

        double totalLabelSum = 0;
        for (int i : dpIdxList)
          totalLabelSum += trainingData[i][labeIdx];

        double yesLabelSum = 0, noLabelSum = 0;

        for (int index = 0; index < dpIdxList.size() - 1; index++) {
          int i = dpIdxList.get(index);
          double label = trainingData[i][labeIdx];
          yesLabelSum += label;
          noLabelSum = totalLabelSum - yesLabelSum;

          int totalSize = dpIdxList.size();
          int yesSize = index + 1;
          int noSize = totalSize - yesSize;

          double delta =
              Math.pow(yesLabelSum, 2) / yesSize + Math.pow(noLabelSum, 2) / noSize
                  - Math.pow(totalLabelSum, 2) / totalSize;


          if (delta > maxDelta) {

            /*
             * System.out.printf(
             * "yesSum %f size: %d \t noSum %f size %d\t totalSum %f  delta %f \t maxdelta %f \n",
             * yesLabelSum, yesSize, noLabelSum, noSize, totalLabelSum, delta, maxDelta);
             */
            maxDelta = delta;
            bestFeatIndex = j;
            bestThreashRow = i;
          }
        }
      }

      return new Pair<Integer, Double>(bestFeatIndex, trainingData[bestThreashRow][bestFeatIndex]);
    }

    @Override
    // find average.
    double decideLabel(List<Integer> dpIdxList) {
      double sum = 0;
      for (int idx : dpIdxList)
        sum += trainingData[idx][labeIdx];

      return sum / dpIdxList.size();
    }

  }

  public RegressionTreeAlgo(IMatrixLoader ld, int labelIdx) {
    super(ld, labelIdx);
  }

  public static void main(String[] s) {
    final int labelIndex = 13;
    HousingMatrixLoader ld = new HousingMatrixLoader();

    RegressionTreeAlgo algo = new RegressionTreeAlgo(ld, labelIndex);
    algo.exec();


  }

}
