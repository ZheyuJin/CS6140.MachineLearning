package dt.dataloader;

import java.util.Random;

public class SamplingSpamLoader implements IMatrixLoader {
  IMatrixLoader ld;
  Random rand = new Random();

  public SamplingSpamLoader() {
    ld = new SpamMatrixLoader();
  }

  @Override
  public double[][] loadTestMatrix() {
    return ld.loadTestMatrix();
  }

  @Override
  public double[][] loadTraningMatrix() {
    double[][] cache = ld.loadTraningMatrix();
    int len = cache.length;
    
    
    /*sampling with replacement*/
    double[][] samples = new double[len][];
    for(int i = 0; i< len; i++){
      int idx = rand.nextInt(len);
      samples[i]= cache[idx];
    }
    
    return samples;
  }

}
