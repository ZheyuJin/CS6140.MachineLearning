package dt.dataloader;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;

public class SpamMatrixLoader implements IMatrixLoader {

  static final int lineLen = 58;
  static final String file = "spam.txt";
  static final String dirctory = "E:\\study\\ML";

  double[][] data;
  static int kfold = 10;
  int curTestFold;

  public SpamMatrixLoader() {
    curTestFold = 0;

    loadAll();
    shuffle();
  }



  // shuffle the data, since it looks like sorted by label.
  private void shuffle() {
    int size = data.length;
    double[] tmp = null;

    Random r = new Random(12);
    for (int i = 0; i < size; i++) {
      int to = r.nextInt(size);
      tmp = data[to];
      data[to] = data[i];
      data[i] = tmp;
    }
  }

  private void loadAll() {
    Path p = FileSystems.getDefault().getPath(dirctory, file);
    List<String> lines = null;
    try {
      lines = Files.readAllLines(p);
    } catch (IOException e) {
      e.printStackTrace();
    }

    assert (lines != null);
    data = new double[lines.size()][lineLen];

    for (int i = 0; i < lines.size(); i++) {
      String line = lines.get(i);
      StringTokenizer st = new StringTokenizer(line, ",");

      for (int j = 0; j < lineLen; j++) {
        data[i][j] = Double.valueOf(st.nextToken());
      }
    }
  }


  boolean trainingLoaded = false;
  boolean testLoaded = false;

  private void tryNextFold() {
    if (trainingLoaded == true && testLoaded == true) {
      trainingLoaded = testLoaded = false;
      curTestFold = (curTestFold + 1) % kfold;
    }
  }

  @Override
  public double[][] loadTraningMatrix() {
    trainingLoaded = true;
    tryNextFold();

    List<double[]> holder = new ArrayList<>();
    for (int index = 0; index < data.length; index++) {
      if (index % kfold != curTestFold)
        holder.add(data[index]);
    }

    double[][] ret = new double[holder.size()][];
    int index = 0;
    for (double[] row : holder)
      ret[index++] = row;

    return ret;
  }


  @Override
  public double[][] loadTestMatrix() {
    testLoaded = true;
    tryNextFold();

    List<double[]> holder = new ArrayList<>();
    for (int index = 0; index < data.length; index++) {
      if (index % kfold == curTestFold)
        holder.add(data[index]);
    }

    double[][] ret = new double[holder.size()][];
    int index = 0;
    for (double[] row : holder)
      ret[index++] = row;

    return ret;
  }

  /**
   * test
   */
  public static void main(String[] args) {
    SpamMatrixLoader ld = new SpamMatrixLoader();
    for (int i = 0; i <= 100; i++) {
      double[][] test = ld.loadTestMatrix();
      double[][] train = ld.loadTraningMatrix();
    }
  }

  private static void noDuplicate(double[][] train, double[][] test) {
    HashSet<double[]> set = new HashSet<>();
    for (double[] d : train) {
      if (set.contains(d)) {
        for (double dd : d) {
          System.err.printf("%f,", dd);
        }
      } else
        set.add(d);
    }

    for (double[] d : test) {
      if (set.contains(d)) {
        for (double dd : d) {
          System.err.printf("%f,", dd);
        }
        System.err.println();
      } else
        set.add(d);
    }

    assert train.length + test.length == set.size() : train.length + " " + test.length + " "
        + set.size();

  }
}
