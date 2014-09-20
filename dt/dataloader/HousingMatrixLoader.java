package dt.dataloader;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;


public class HousingMatrixLoader implements IMatrixLoader {

  static final int lineLen = 14;
  static final String trainFile = "housing_train.txt";
  static final String testFile = "housing_test.txt";
  static final String dirctory = "E:\\study\\ML";

  private double[][] train;
  private double[][] test;


  double[][] readData(String dir, String file) {
    Path p = FileSystems.getDefault().getPath(dir, file);
    List<String> lines = null;
    try {
      lines = Files.readAllLines(p);
    } catch (IOException e) {
      e.printStackTrace();
    }

    double[][] ret = new double[lines.size()][lineLen];


    for (int i = 0; i < lines.size(); i++) {
      String line = lines.get(i);
      Scanner sc = new Scanner(line);
      if (!sc.hasNextDouble())
        break;

      // System.err.println(lines.get(i));
      for (int j = 0; j < lineLen; j++) {
        ret[i][j] = sc.nextDouble();
      }
      sc.close();
    }

    return ret;
  }

  @Override
  public double[][] loadTestMatrix() {
    test = readData(dirctory, testFile);
    return test;
  }

  @Override
  public double[][] loadTraningMatrix() {
    train = readData(dirctory, trainFile);
    return train;
  }


  /**
   * @param args
   */
  public static void main(String[] args) {
    HousingMatrixLoader ld = new HousingMatrixLoader();
    double[][] d = ld.loadTestMatrix();
    for (double[] dd : d)
      System.out.println(dd[13]);
  }

}
