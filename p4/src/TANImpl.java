import java.util.*;
import weka.core.*;
import java.io.*;

public class TANImpl{
	private Instances trainData = null;
	public int numAttr = 0;
	Double[][] cmiMatrix = null;
	AttrValuesPair[][] probXiXjY = null;
	AttrValuesPair[][] probXiXjGivenY = null;
	public String label1 = null;
	public String label2 = null;

	public ArrayList<ArrayList<Double>> label1ProbList = null;
	public ArrayList<ArrayList<Double>> label2ProbList = null;

	public HashSet<Integer> inTree = null;
	public Integer[][] treeEdge = null;

	public ArrayList<MSTNode> treeNodeList = null;



	public TANImpl(Instances trainData) {
		this.trainData = trainData;
		this.numAttr = trainData.numAttributes()-1;
		this.cmiMatrix = new Double[numAttr][numAttr];
		this.probXiXjY = new AttrValuesPair[numAttr][numAttr];
		this.probXiXjGivenY = new AttrValuesPair[numAttr][numAttr];
		this.label1 = new String(trainData.attribute(numAttr).value(0));
		this.label2 = new String(trainData.attribute(numAttr).value(1));

		this.label1ProbList = new ArrayList<>();
		this.label2ProbList = new ArrayList<>();

		this.inTree = new HashSet<Integer>();
		this.treeNodeList = new ArrayList<MSTNode>();
	}

	public void train() {
		computeProbXiXjY(this.probXiXjY, this.probXiXjGivenY);
		computeProbXGivenY(this.label1ProbList, this.label2ProbList);
		computeCMI();
		MSTNode mstRoot = new MSTNode(trainData.attribute(0).name(), trainData.attribute(numAttr).name());
		this.inTree.add(0);
		this.treeEdge = new Integer[2][numAttr];
		for (int i = 0; i < numAttr; i++) {
			treeEdge[1][i] = i;
		}
		
		while (inTree.size() < numAttr) {
			double maxCMI = 0.0;
			int treeParentIndex = -1;
			int newTreenodeIndex = -1;
			for (Integer i:inTree) {
				for (int j = 0; j < numAttr; j++) {
					//System.out.printf("%.6f\n",this.cmiMatrix[Integer.valueOf(i)][j]);
					if (this.cmiMatrix[Integer.valueOf(i)][j] > maxCMI && !inTree.contains(Integer.valueOf(j))) {
						newTreenodeIndex = j;
						treeParentIndex = i;
						maxCMI = this.cmiMatrix[i][j];
					}
				}
			}
			inTree.add(newTreenodeIndex);
			this.treeEdge[0][newTreenodeIndex] = treeParentIndex;
		}
		/* print dad
		for (int i = 0; i < numAttr; i++) {
			System.out.println("pair: child is " + this.treeEdge[1][i] + " dad is " + this.treeEdge[0][i]);
		} */
		// assign child|dad0, dad1 prob matrix

		for (int i = 0; i < numAttr; i++) {
			if (i == 0) { // first attr
				for (int j = 0; j < trainData.attribute(i).numValues(); j++) {
					Double[][] toAdd = new Double[1][trainData.attribute(numAttr).numValues()];
					computeCondProbDads(toAdd, i, trainData.attribute(i).value(j), numAttr, -1);
					mstRoot.condProbTableList.add(toAdd);
				}
				this.treeNodeList.add(mstRoot);
				//System.out.printf("%.12f\n", mstRoot.condProbTableList.get(2)[0][0]);
			} else {
				MSTNode newNode = new MSTNode(this.trainData.attribute(i).name(), this.trainData.attribute(this.treeEdge[0][i]).name());
				newNode.parents.add(trainData.attribute(numAttr).name()); 
				for (int j = 0; j < trainData.attribute(i).numValues(); j++) {
					int dad1 = this.treeEdge[0][i];
					Double[][] toAdd = new Double[this.trainData.attribute(dad1).numValues()][this.trainData.attribute(numAttr).numValues()];
					computeCondProbDads(toAdd, i, trainData.attribute(i).value(j), dad1, numAttr);
					newNode.condProbTableList.add(toAdd);
				}
				this.treeNodeList.add(newNode);
			}
		}
	}

	public int test(Instances testData) {
		// step 1 print dad
		for (MSTNode node:treeNodeList) {
			System.out.print(node.nodeName);
			for (String dadName:node.parents) {
				System.out.print(" " + dadName);
			}
			System.out.print("\n");
		}
		System.out.print("\n");
		// do prediction
		int numLabel1inTrain = countNumLabel1();
		double probLabel1, probLabel2;
		int correctCount = 0;
		// print CPT
		/*
		for (int i = 1; i < numAttr; i++) {
			int dadIndex = this.treeEdge[0][i];
			for (int j = 0; j < this.trainData.attribute(i).numValues(); j++) {
				for (int k = 0; k < this.trainData.attribute(dadIndex).numValues(); k++) {
					System.out.print("Pr(" + i+ "=" + j + " |" + dadIndex + "=" + k + ", 18=0) = ");
					System.out.printf("%.18f\n", this.treeNodeList.get(i).condProbTableList.get(j)[k][0]);
				}
				
			}
			
		}*/

		for (Instance insta:testData) {
			// reset
			probLabel1 = ((double)(numLabel1inTrain+1))/((double)(this.trainData.numInstances()+2));
			probLabel2 = (double)1 - probLabel1;

			for (int i = 0; i < this.numAttr; i++) {
				if (i == 0) {
					int pos = testData.attribute(i).indexOfValue(insta.toString(i));
					probLabel1 *= this.label1ProbList.get(i).get(pos);
					probLabel2 *= this.label2ProbList.get(i).get(pos);
				} else {

					int dad1Index = this.treeEdge[0][i];
					int dadValIndex = this.trainData.attribute(dad1Index).indexOfValue(insta.toString(dad1Index));
					int attrValIndex = this.trainData.attribute(i).indexOfValue(insta.toString(i));
					//probLabel1 *= computeProbXi(dad1Index, testData.attribute(dad1Index).indexOfValue(insta.toString(dad1Index)));
					//probLabel2 *= computeProbXi(dad1Index, testData.attribute(dad1Index).indexOfValue(insta.toString(dad1Index)));
					double tmp1 = this.treeNodeList.get(i).condProbTableList.get(attrValIndex)[dadValIndex][0];
					probLabel1 *= tmp1;
					double tmp2 = this.treeNodeList.get(i).condProbTableList.get(attrValIndex)[dadValIndex][1];
					probLabel2 *= tmp2;
					//System.out.print("probLabel1 after " + i + " th attribute, of val Index: " + attrValIndex + " whose dad index is: " + dad1Index + "   dad value index" + dadValIndex + "   ");
					//System.out.printf("%.18f\n", tmp1);
				}
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

	private double computeProbXi(int attrIndex, int attrValIndex) {
		int hitCount = 0;
		for (Instance insta:this.trainData) {
			if (this.trainData.attribute(attrIndex).indexOfValue((insta.toString(attrIndex))) == attrValIndex) hitCount++;
		}
		return (((double)(hitCount+1))/(double)(this.trainData.numInstances()+this.trainData.attribute(attrIndex).numValues()));
	}

	private void computeCondProbDads(Double[][] probMatrix, int childAttr, String childVal, int dad1, int dad2) {
		if (dad2 == -1) { // root case
			int countlabel1 = 0, countlabel2 = 0;
			for (Instance insta:this.trainData) {
				if (insta.toString(childAttr).equals(childVal)) {
					if (insta.toString(numAttr).equals(this.label1)) countlabel1++;
					else countlabel2++;
				}
			}
			int demoninator1 = countNumLabel1() + this.trainData.attribute(childAttr).numValues();
			int demoninator2 = this.trainData.numInstances() - countNumLabel1() + this.trainData.attribute(childAttr).numValues();
			probMatrix[0][0] = ((double)countlabel1+1)/((double)(demoninator1));
			probMatrix[0][1] = ((double)countlabel2+1)/((double)(demoninator2));
			/*
			System.out.print("Root index: " + childAttr + " child val index: " + childVal + " and dad1 val = label1   " );
			System.out.printf("%.18f\n", probMatrix[0][0]);
			System.out.print("Root index: " + childAttr + " child val index: " + childVal + " and dad1 val = label2   " );
			System.out.printf("%.18f\n", probMatrix[0][1]);
			*/
		} else {
			for (int i = 0; i < probMatrix.length; i++) {
				int countlabel1 = 0, countlabel2 = 0;
				int demoninator1 = 0, demoninator2 = 0;
				for (Instance insta:this.trainData) {
					if (insta.toString(childAttr).equals(childVal)) {
						if (insta.toString(dad1).equals(trainData.attribute(dad1).value(i))) {
							if (insta.toString(numAttr).equals(this.label1)) countlabel1++;
							else countlabel2++;
						}
					}
					if (insta.toString(dad1).equals(trainData.attribute(dad1).value(i))) {
						if (insta.toString(numAttr).equals(this.label1)) demoninator1++;
						else demoninator2++;
					}
				}
				//if ((demoninator1 + demoninator2) != trainData.numInstances()) System.out.println ("WRONG COUNT !!!!!!!");
				demoninator1 += this.trainData.attribute(childAttr).numValues();
				demoninator2 += this.trainData.attribute(childAttr).numValues();
				probMatrix[i][0] = ((double)countlabel1+1)/((double)(demoninator1));
				probMatrix[i][1] = ((double)countlabel2+1)/((double)(demoninator2));
				/*
				System.out.print("child index: " + childAttr + " dad1 index: " + dad1 + " dad1 val index: " + i + "and dad2 val = 0   " );
				System.out.printf("%.18f\n", probMatrix[i][0]);
				System.out.print("child index: " + childAttr + " dad2 index: " + dad1 + " dad1 val index: " + i + " and dad2 val = 1  " );
				System.out.printf("%.18f\n", probMatrix[i][1]);
				*/
			}
		}
	}

	/**
	 * Helper method for computing conditional mutual information
	 */
	private void computeProbXGivenY(ArrayList<ArrayList<Double>> label1ProbList, ArrayList<ArrayList<Double>> label2ProbList) {
		for (int i = 0; i < this.numAttr; i++) {
			int[] label1ValCount = new int[trainData.attribute(i).numValues()];
			int[] label2ValCount = new int[trainData.attribute(i).numValues()];
			for (Instance insta:this.trainData) {
				int pos = this.trainData.attribute(i).indexOfValue(insta.toString(i));
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
	/**
	 * Helper method for computing conditional mutual information
	 */
	private void computeProbXiXjY(AttrValuesPair[][] probXiXjY, AttrValuesPair[][] probXiXjGivenY) {
		for (int i = 0; i < numAttr; i++) {
			for (int j = 0; j < numAttr; j++) {
				int numValueXi = trainData.attribute(i).numValues(), numValueXj = trainData.attribute(j).numValues();
				AttrValuesPair toAdd = new AttrValuesPair(numValueXi, numValueXj, trainData.attribute(numAttr).numValues(), trainData.numInstances());
				AttrValuesPair toAddGivenY = new AttrValuesPair(numValueXi, numValueXj, trainData.attribute(numAttr).numValues(), trainData.numInstances());
				for (Instance insta:trainData) {
					int pos_i = trainData.attribute(i).indexOfValue(insta.toString(i));
					int pos_j = trainData.attribute(j).indexOfValue(insta.toString(j));
					if (insta.toString(numAttr).equals(this.label1)) {
						toAdd.label1Table[pos_i][pos_j] += 1;
						toAddGivenY.label1Table[pos_i][pos_j] += 1;
					} else {
						toAdd.label2Table[pos_i][pos_j] += 1;
						toAddGivenY.label2Table[pos_i][pos_j] += 1;
					}
				}
				toAdd.calcProb();
				/*
				for (int k = 0; k < trainData.attribute(i).numValues(); k++) {
					for (int m = 0; m < trainData.attribute(j).numValues(); m++) {
						System.out.printf("%.6f\n", toAdd.label1Table[k][m]);
					}
				}*/
				toAddGivenY.calcCondProb(countNumLabel1());
				probXiXjY[i][j] = toAdd; 
				probXiXjGivenY[i][j] = toAddGivenY;
				/*
				for (int k = 0; k < trainData.attribute(i).numValues(); k++) {
					for (int m = 0; m < trainData.attribute(j).numValues(); m++) {
						System.out.printf("%.18f\n", toAddGivenY.label1Table[k][m]);
					}
				}*/
			}
		}
	}

	private int countNumLabel1() {
		int count = 0;
		for (Instance insta:this.trainData) {
			if (insta.stringValue(numAttr).equals(this.label1)) count++;
		}
		return count;
	}
	/**
	 * Helper method for computing conditional mutual information
	 */
	private void computeCMI() {

		for (int i = 0; i < numAttr; i++) {
			for (int j = i+1; j < numAttr; j++) {
				double tmp = 0;
				for (int ival = 0; ival < this.trainData.attribute(i).numValues(); ival++) {
					for (int jval = 0; jval < this.trainData.attribute(j).numValues(); jval++) {
						double label1InLog = probXiXjGivenY[i][j].label1Table[ival][jval]/(label1ProbList.get(i).get(ival) * label1ProbList.get(j).get(jval));
						double label2InLog = probXiXjGivenY[i][j].label2Table[ival][jval]/(label2ProbList.get(i).get(ival) * label2ProbList.get(j).get(jval));
						tmp += (probXiXjY[i][j].label1Table[ival][jval]*Math.log(label1InLog)/Math.log(2.0) + probXiXjY[i][j].label2Table[ival][jval]*Math.log(label2InLog)/Math.log(2.0));
					}
					//System.out.printf("%.6f\n", tmp);
				}
				this.cmiMatrix[i][j] = tmp;
			}
		}
		for (int i = 0; i < numAttr; i++) {
			this.cmiMatrix[i][i] = -1.0;
		}
		// finish the symmetric part for easier calc
		for (int i = 1; i < numAttr; i++) {
			for (int j = 0; j < i; j++) {
				this.cmiMatrix[i][j] = this.cmiMatrix[j][i];
				//System.out.printf("%.12f\n", this.cmiMatrix[j][i]);
			}
		}
		/*
		for (int i = 0; i < numAttr; i++) {
			for (int j = 0; j < numAttr; j++) {
				System.out.printf("%.12f", this.cmiMatrix[j][i]);
				System.out.print(" ");
			}
			System.out.print("\n");
		}
		*/
	}
}