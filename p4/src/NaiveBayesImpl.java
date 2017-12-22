import java.util.*;
import weka.core.*;
import java.io.*;

public class NaiveBayesImpl {
	private Instances trainData = null;
	public ArrayList<ArrayList<Double>> label1ProbList = null;
	public ArrayList<ArrayList<Double>> label2ProbList = null;
	public int numAttr = 0;
	public String label1 = null;
	public String label2 = null;

	public NaiveBayesImpl(Instances trainData) {
		this.trainData = trainData;
		this.numAttr = trainData.numAttributes()-1;
		this.label1 = new String(trainData.attribute(numAttr).value(0));
		this.label2 = new String(trainData.attribute(numAttr).value(1));
		label1ProbList = new ArrayList<>();
		label2ProbList = new ArrayList<>();
	}

	public void train() {
		for (int i = 0; i < this.numAttr; i++) {
			int[] label1ValCount = new int[trainData.attribute(i).numValues()];
			int[] label2ValCount = new int[trainData.attribute(i).numValues()];
			for (Instance insta:trainData) {
				int pos = trainData.attribute(i).indexOfValue(insta.toString(i));
				if (insta.toString(numAttr).equals(this.label1)) {
					label1ValCount[pos] += 1;
				} else if (insta.toString(numAttr).equals(this.label2)) {
					label2ValCount[pos] += 1;
				} else {
					System.out.println("Error in NB train: class label error");
					System.exit(-1);
				}
			}
			int totalLabel1 = 0, totalLabel2 = 0;
			//int tmpTotalnum = trainData.numInstances();
			for (int j = 0; j < label1ValCount.length; j++) {
				totalLabel1 = totalLabel1 + label1ValCount[j] + 1;
				totalLabel2 = totalLabel2 + label2ValCount[j] + 1;
				//tmpTotalnum = tmpTotalnum - label1ValCount[j] - label2ValCount[j];
			}
			//System.out.println("tmpTotalnum is: " + tmpTotalnum);
			ArrayList<Double> toAdd1 = new ArrayList<Double>();
			ArrayList<Double> toAdd2 = new ArrayList<Double>();
			for (int j = 0; j < label1ValCount.length; j++) {
				//System.out.println("number of label1ValCount[j]: " + label1ValCount[j] +", and totalLabel1: " + totalLabel1 + " current attribute index: " + i);
				//System.out.println("number of label2ValCount[j]: " + label2ValCount[j] +", and totalLabel2: " + totalLabel2 + " current attribute index: " + i);
				toAdd1.add(((double)(label1ValCount[j]+1))/((double)totalLabel1));
				toAdd2.add(((double)(label2ValCount[j]+1))/((double)totalLabel2));
			}
			label1ProbList.add(new ArrayList<Double>(toAdd1));
			label2ProbList.add(new ArrayList<Double>(toAdd2));
		}
	}

	public int test(Instances testData) {
		double probLabel1, probLabel2;
		int countLabel1 = 0;
		for (Instance insta:this.trainData) {
			if (insta.toString(numAttr).equals(this.label1)) countLabel1++;
		}
		int correctCount = 0;
		for (Instance insta:testData) {
			// reset
			probLabel1 = ((double)(countLabel1+1))/((double)(this.trainData.numInstances()+2));
			probLabel2 = (double)1 - probLabel1;

			for (int i = 0; i < numAttr; i++) {
				int pos = testData.attribute(i).indexOfValue(insta.toString(i));
				probLabel1 *= this.label1ProbList.get(i).get(pos);
				probLabel2 *= this.label2ProbList.get(i).get(pos);
			}

			if (probLabel1 >= probLabel2) {
				System.out.print(this.label1 + " " + insta.toString(numAttr) + " ");
				System.out.printf("%.12f\n", probLabel1/(probLabel1+probLabel2));
				if (this.label1.equals(insta.toString(numAttr))) correctCount++;
			} else {
				System.out.print(this.label2 + " " + insta.toString(numAttr) + " ");
				System.out.printf("%.12f\n", probLabel2/(probLabel1+probLabel2));
				if (this.label2.equals(insta.toString(numAttr))) correctCount++;
			}
		}
		System.out.print("\n" + correctCount);
		return correctCount;
	}
}




