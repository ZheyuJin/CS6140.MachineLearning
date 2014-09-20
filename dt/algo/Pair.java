package dt.algo;

public class Pair<K, V> {
  private final K first;
  private final V second;

  public Pair(K one, V two) {
    this.first = one;
    this.second = two;
  }

  public K getFirst() {
    return first;
  }

  public V getSecond() {
    return second;
  }
}
