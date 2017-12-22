public class FoldPredConf {
	private int fold = -1; // the index of fold
	private String predictLabel; // the predict label
	private Double confidence; // the output value

	/**
	 * Constructor
	 */
	public FoldPredConf(int fold, String predictLabel, Double confidence) {
		this.fold = fold;
		this.predictLabel = predictLabel;
		this.confidence = confidence;
	}

	public int getFold() {
		return this.fold;
	}

	public String getPredLabel() {
		return this.predictLabel;
	}

	public Double getConf() {
		return this.confidence;
	}
}