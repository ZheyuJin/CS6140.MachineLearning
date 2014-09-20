package dt.algo;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import dt.dataloader.IMatrixLoader;


/**
 * abstract class for algorithm. Some logic is left for subclases to fill.
 */
public abstract class Algo {

  IMatrixLoader loader;
  DTree tree;
  int labelIdx;

  double[][] trainMatrix;
  double[][] testMatrix;

  Algo(IMatrixLoader ld, int labelIndex) {
    loader = ld;
    labelIdx = labelIndex;
  }

  protected void init() {
    trainMatrix = loader.loadTraningMatrix();
    testMatrix = loader.loadTestMatrix();

    assert trainMatrix != null;
    assert testMatrix != null;
  }


  abstract void exec();

  /**
   * abstract class for decision tree. Some logics are left for subclasses to fill.
   */
  abstract class DTree {
    Node root;

    int labeIdx;
    double NODE_SIZE_LIMIT_RATIO;

    List<Node> leafList = new ArrayList<>();

    int nodeCount;
    int leafCount;

    double[][] trainingData;
    double[][] testData;

    DTree(double[][] train, double[][] test, int labelIdx, double ratio) {
      trainingData = train;
      testData = test;
      labeIdx = labelIdx;
      NODE_SIZE_LIMIT_RATIO = ratio;
    }

    private boolean shouldStopSplit(List<Integer> dpIdxList) {
      return trainingData.length * NODE_SIZE_LIMIT_RATIO >= dpIdxList.size();
    }

    // returns label of terminal node
    // TODO
    double test(double[] dp) {
      Node cur = root;

      while (false == cur.isLeaf()) {
        if (isLeft(cur.thersh, dp[cur.featIdx])) {
          cur = cur.left;
        } else
          cur = cur.right;
      }

      return cur.lable;
    }

    void train() {
      ArrayList<Integer> lst = new ArrayList<>();
      for (int i = 0; i < trainingData.length; i++)
        lst.add(i);

      assert (lst.size() != 0);
      root = recursive(lst);
    }


    private Node registerTerminalNode(List<Integer> dpIdxList) {
      double labelOut = decideLabel(dpIdxList);
      Node n = new Node(dpIdxList, labelOut);
      leafCount++;
      nodeCount++;
      leafList.add(n);
      return n;
    }

    /**
     * split rule
     * */
    private boolean isLeft(double threash, double target) {
      return threash >= target;
    }

    private Node recursive(List<Integer> dpIdxList) {
      if (shouldStopSplit(dpIdxList)) // lets stop splitting
        return registerTerminalNode(dpIdxList);

      /* lets split */
      Pair<Integer, Double> ans = bestSplit(dpIdxList);

      int bestFeatIndex = ans.getFirst();
      double bestThresh = ans.getSecond();
      /*
       * System.err.printf("bestsplit %8d \t bestThresh %f \t indexList size %d \n", bestFeatIndex,
       * bestThresh, dpIdxList.size());
       */
      // split data points by theta
      List<Integer> leftList = new ArrayList<>();
      List<Integer> rightList = new ArrayList<>();

      for (int idx : dpIdxList) {
        if (isLeft(bestThresh, trainingData[idx][bestFeatIndex])) // smallest value goes to left.
          leftList.add(idx);
        else
          rightList.add(idx);
      }

      /*
       * if there is a thresh so that all data goes into one branch, then no need to split. make me
       * terminal.
       */
      if (0 == leftList.size() || 0 == rightList.size())
        return registerTerminalNode(dpIdxList);

      Node n = new Node(bestFeatIndex, bestThresh);
      nodeCount++;
      n.left = recursive(leftList);
      n.right = recursive(rightList);
      return n;
    }

    // TODO
    abstract Pair<Integer, Double> bestSplit(List<Integer> dpIdxList);


    // TODO
    abstract double decideLabel(List<Integer> dpIdxList);


  }

  class Node {
    /* only meaningful for terminal node. */
    double lable;
    List<Integer> tdpIdxList;

    /* only meaningful for non-terminal node */
    double thersh;
    Node left;
    Node right;
    int featIdx; // feature for split

    // non - leaf constructor
    Node(int featIndex, double t) {
      featIdx = featIndex;
      thersh = t;
    }

    // leaf constructor
    Node(List<Integer> list, double lab) {
      tdpIdxList = list;
      lable = lab;
    }

    boolean isLeaf() {
      return left == null && right == null;
    }

    int getSize() {
      if (tdpIdxList == null)
        return 0;

      return tdpIdxList.size();
    }
  }

}
