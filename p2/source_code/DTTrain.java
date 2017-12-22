import weka.core.*;
import java.util.*;

public class DTTrain {
	private int m;
	public DTNode treeRoot;

	public DTTrain(Instances data, int m) {
		this.m = m;
		this.treeRoot = makeRoot(data, commonLabel(data, "none"));
	}

	private DTNode makeRoot(Instances data, String common) {
		double infoGain = Integer.MIN_VALUE;
		Attribute rootAttr = null;
		// calculate the total entropy H(D)
		double totalEntropy = calTotalEntropy(data);
		double thred = 0.000000000000;
		for (int i = 0; i < data.numAttributes()-1; i++) {
			if (data.attribute(i).type() == Attribute.NOMINAL) {
				if (nominalCalcInfoGain(data, i, totalEntropy) - infoGain > 10e-7) {
					infoGain = nominalCalcInfoGain(data, i, totalEntropy);
					rootAttr = data.attribute(i);
				}				
				
			} else if (data.attribute(i).type() == Attribute.NUMERIC) {
				double[] candiThred = determineCandidateNumericSplits(data, i);
				for (int j = 0; j < candiThred.length; j++) {
					if (numericCalcInfoGain(data, i, candiThred[j], totalEntropy) - infoGain > 10e-7) {
						infoGain = numericCalcInfoGain(data, i, candiThred[j], totalEntropy);
						rootAttr = data.attribute(i);
						thred = candiThred[j];
					} 
				}
			} else {
				System.err.println("Error: feature is neither nominal nor numeric");
				System.exit(-1);
			}
		}

		DTNode rootNode = new DTNode(false, rootAttr, "NA", Integer.MAX_VALUE, -1, common);
		ArrayList<DTNode> childrenList = new ArrayList<DTNode>();
		ArrayList<ArrayList<Integer>> numLabeledInstances = new ArrayList<ArrayList<Integer>>();
		if (rootAttr.type() == Attribute.NOMINAL) {
			for (int i = 0; i < rootAttr.numValues(); i++) {
				Instances subInstances = nominalFindSubData(data, rootAttr, rootAttr.value(i));
				DTNode child = makeSubtree(subInstances, commonLabel(subInstances, common), rootAttr, rootAttr.value(i), Integer.MAX_VALUE, -1);
				childrenList.add(child);
				ArrayList<Integer> numToAdd = new ArrayList<Integer>();
				numToAdd.add(calNumFirstLabel(subInstances));
				numToAdd.add(subInstances.numInstances() - calNumFirstLabel(subInstances));
				numLabeledInstances.add(numToAdd);
			}
			rootNode.children = childrenList;
			rootNode.numInstances = numLabeledInstances;
		} else if (rootAttr.type() == Attribute.NUMERIC) {
			Instances left = numericFindLeftData(data, rootAttr, thred);
			Instances right = numericFindRightData(data, rootAttr, thred);
			DTNode leftChild = makeSubtree(left, common, rootAttr, "NA", thred, 0);
			DTNode rightChild = makeSubtree(right, common, rootAttr, "NA", thred, 1);
			childrenList.add(leftChild);
			childrenList.add(rightChild);
			ArrayList<Integer> leftNum = new ArrayList<Integer>();
			leftNum.add(calNumFirstLabel(left));
			leftNum.add(left.numInstances() - calNumFirstLabel(left));
			ArrayList<Integer> rightNum = new ArrayList<Integer>();
			rightNum.add(calNumFirstLabel(right));
			rightNum.add(right.numInstances() - calNumFirstLabel(right));
			numLabeledInstances.add(leftNum);
			numLabeledInstances.add(rightNum);
			rootNode.children = childrenList;
			rootNode.numInstances = numLabeledInstances;
		}
		return rootNode;
	}

	private DTNode makeSubtree(Instances data, String common, Attribute currAttr,
							   String attrValue, double currThreshold, int side) { // side is ued for determine whether larger or less
		// find the candidate splits
		int numCandiSplit = 0;

		double infoGain = Integer.MIN_VALUE;
		Attribute rootAttr = null;
		// calculate the total entropy H(D)
		double totalEntropy = calTotalEntropy(data);
		double thred = 0.000000000000;
		for (int i = 0; i < data.numAttributes()-1; i++) {

			if (data.attribute(i).type() == Attribute.NOMINAL) {
				numCandiSplit++;
				if (nominalCalcInfoGain(data, i, totalEntropy) - infoGain > 10e-7) {
					infoGain = nominalCalcInfoGain(data, i, totalEntropy);
					rootAttr = data.attribute(i);
				}

			} else if (data.attribute(i).type() == Attribute.NUMERIC) {
				double[] candiThred = determineCandidateNumericSplits(data, i);
				if (candiThred.length > 0) numCandiSplit++;
				for (int j = 0; j < candiThred.length; j++) {
					if (numericCalcInfoGain(data, i, candiThred[j], totalEntropy) - infoGain > 10e-7) {
						infoGain = numericCalcInfoGain(data, i, candiThred[j], totalEntropy);
						rootAttr = data.attribute(i);
						thred = candiThred[j];
					} 
				}

			} else {
				System.err.println("Error: feature is neither nominal nor numeric");
				System.exit(-1);
			}
		}

		// check if meet stopping criteria
		if (data.numInstances() == 0) {
			DTNode leaf = new DTNode(true, currAttr, attrValue, currThreshold, side, common);
			return leaf;
		}
		if (allSameLabel(data)) {
			// make leaf node, and determine label
			String label = data.instance(0).toString(data.numAttributes()-1);
			DTNode leaf = new DTNode(true, currAttr, attrValue, currThreshold, side, label);
			return leaf;
		}
		if (data.numInstances() < this.m || infoGain <= 0 || numCandiSplit <= 0) {
			int first = calNumFirstLabel(data);
			int second = data.numInstances() - first;
			DTNode leaf;
			if (first > second) {
				String label = data.attribute(data.numAttributes()-1).value(0);
				leaf =  new DTNode(true, currAttr, attrValue, currThreshold, side, label);	
			} else if (first == second) {
				leaf = new DTNode(true, currAttr, attrValue, currThreshold, side, common);
			} else {
				String label = data.attribute(data.numAttributes()-1).value(1);
				leaf = new DTNode(true, currAttr, attrValue, currThreshold, side, label);
			}
			return leaf;
		}

		// else, make an internal node
		DTNode internalNode = new DTNode(false, currAttr, attrValue, currThreshold, side, common);
		ArrayList<DTNode> childrenList = new ArrayList<DTNode>();
		ArrayList<ArrayList<Integer>> numLabeledInstances = new ArrayList<ArrayList<Integer>>();
		if (rootAttr.type() == Attribute.NOMINAL) {
			for (int i = 0; i < rootAttr.numValues(); i++) {
				Instances subInstances = nominalFindSubData(data, rootAttr, rootAttr.value(i));
				DTNode child = makeSubtree(subInstances, commonLabel(subInstances, common), rootAttr, rootAttr.value(i), Integer.MAX_VALUE, -1);
				childrenList.add(child);
				ArrayList<Integer> numToAdd = new ArrayList<Integer>();
				numToAdd.add(calNumFirstLabel(subInstances));
				numToAdd.add(subInstances.numInstances() - calNumFirstLabel(subInstances));
				numLabeledInstances.add(numToAdd);
				
			}
			internalNode.children = childrenList;
			internalNode.numInstances = numLabeledInstances;
		} else if (rootAttr.type() == Attribute.NUMERIC) {
			Instances left = numericFindLeftData(data, rootAttr, thred);
			Instances right = numericFindRightData(data, rootAttr, thred);
			String leftcommon = commonLabel(left, common);
			String rightcommon = commonLabel(right, common);
			DTNode leftChild = makeSubtree(left, leftcommon, rootAttr, "NA", thred, 0);
			DTNode rightChild = makeSubtree(right, rightcommon, rootAttr, "NA", thred, 1);
			childrenList.add(leftChild);
			childrenList.add(rightChild);
			ArrayList<Integer> leftNum = new ArrayList<Integer>();
			leftNum.add(calNumFirstLabel(left));
			leftNum.add(left.numInstances() - calNumFirstLabel(left));
			ArrayList<Integer> rightNum = new ArrayList<Integer>();
			rightNum.add(calNumFirstLabel(right));
			rightNum.add(right.numInstances() - calNumFirstLabel(right));
			numLabeledInstances.add(leftNum);
			numLabeledInstances.add(rightNum);
			internalNode.children = childrenList;
			internalNode.numInstances = numLabeledInstances;
			
		}
		return internalNode;
	}

	private int calNumFirstLabel(Instances data) {
		int total = data.numInstances();
		int numFirstLabel = 0;
		String firstLabel = data.attribute(data.numAttributes()-1).value(0);
		for (int i = 0; i < total; i++) {
			if (data.instance(i).toString(data.numAttributes()-1).equals(firstLabel)) numFirstLabel++;
		}
		return numFirstLabel;
	}

	private Instances nominalFindSubData(Instances allData, Attribute attr, String value) {
		Instances result = new Instances(allData);
		result.delete();
		for (int i = 0; i < allData.numInstances(); i++) {
			if (allData.instance(i).stringValue(attr).equals(value)) {
				result.add(allData.instance(i));
			}
		}
		return result;
	}

	private Instances numericFindLeftData(Instances allData, Attribute attr, double threshold) {
		Instances result = new Instances(allData);
		result.delete();
		for (int i = 0; i < allData.numInstances(); i++) {
			if (allData.instance(i).value(attr) <= threshold) {
				result.add(allData.instance(i));
			}
		}
		return result;
	}

	private Instances numericFindRightData(Instances allData, Attribute attr, double threshold) {
		Instances result = new Instances(allData);
		result.delete();
		for (int i = 0; i < allData.numInstances(); i++) {
			if (allData.instance(i).value(attr) > threshold) {
				result.add(allData.instance(i));
			}
		}
		return result;
	}

	private double calTotalEntropy (Instances data) {
		int total = data.numInstances();
		if (total == 0) return 0.000000000000;
		int numFirstLabel = 0;
		String firstLabel = data.attribute(data.numAttributes()-1).value(0);
		for (int i = 0; i < total; i++) {
			if (data.instance(i).toString(data.numAttributes()-1).equals(firstLabel)) numFirstLabel++;
		}
		int numSecondLabel = total - numFirstLabel;
		double tmp1 = (double)numFirstLabel/(double)total;
		double tmp2 = (double)numSecondLabel/(double)total;
		if (numFirstLabel == 0 || numSecondLabel == 0) return 0.000000000000;
		return -tmp1*(Math.log(tmp1)/Math.log(2.000000000000)) - tmp2*(Math.log(tmp2)/Math.log(2.000000000000));
	}

	private String commonLabel(Instances data, String preCommon) {
		int total = data.numInstances();
		int numFirstLabel = 0;
		String firstLabel = data.attribute(data.numAttributes()-1).value(0);
		String secondLabel = data.attribute(data.numAttributes()-1).value(1);
		for (int i = 0; i < total; i++) {
			if (data.instance(i).toString(data.numAttributes()-1).equals(firstLabel)) numFirstLabel++;
		}
		int numSecondLabel = total - numFirstLabel;
		if (numFirstLabel > numSecondLabel) return firstLabel;
		else if (numFirstLabel == numSecondLabel) return preCommon;
		else return secondLabel;
	}

	private boolean allSameLabel(Instances data) {
		if (data.numInstances() == 0) return true;
		String label = data.firstInstance().stringValue(data.numAttributes()-1);
		for (int i = 1; i < data.numInstances(); i++) {
			if (!data.instance(i).stringValue(data.numAttributes()-1).equals(label)) return false;
		}
		return true;
	}

	private double[] determineCandidateNumericSplits(Instances data, int attrIndex) {
		double[] candi = new double[data.numInstances()];
		
		// Partition
		HashMap<Double, ArrayList<Instance>> map = new HashMap<>();
		for (int i = 0; i < data.numInstances(); i++) {
			if (!map.containsKey(data.instance(i).value(attrIndex))) {
				ArrayList<Instance> toAdd = new ArrayList<Instance>();
				toAdd.add(data.instance(i));
				map.put(data.instance(i).value(attrIndex), toAdd);
			} else {
				map.get(data.instance(i).value(attrIndex)).add(data.instance(i));
			}
		}
		// sort partitions based on value
		Set<Double> set = map.keySet();
		double[] part = new double[set.size()];
		int i = 0;
		for(Double d:set) {
			part[i++] = d;
		}
		//System.out.println(set.size());
		Arrays.sort(part);
		// find and add candidate splits
		int numClass = data.numAttributes()-1; // the index of class label
		int count = 0; // a counter used to count how many candi-thresholds
		for (i = 0; i < part.length-1; i++) {
			ArrayList<Instance> tmpList1 = map.get(part[i]);
			ArrayList<Instance> tmpList2 = map.get(part[i+1]);
			outerloop:
			for (int j = 0; j < tmpList1.size(); j++) {
				for (int k = 0; k < tmpList2.size(); k++) {
					if (!tmpList1.get(j).stringValue(numClass).equals(tmpList2.get(k).stringValue(numClass))) {
						candi[count++] = (part[i] + part[i+1])/2.000000000000;
						break outerloop;
					}
				}
			}
		}
		//System.out.println(count);
		return Arrays.copyOfRange(candi, 0, count);	
	}

	private double nominalCalcInfoGain(Instances data, int attrIndex, double totalEntropy) {
		int numClasses = data.attribute(attrIndex).numValues();
		double[] probX = new double[numClasses];
		double[] entropyYgivenX = new double[numClasses];
		int totalInstances = data.numInstances();
		for (int i = 0; i < numClasses; i++) {
			String currValue = data.attribute(attrIndex).value(i);
			
			int numCurr = 0;
			Instances subInstances = new Instances(data);
			subInstances.delete();
			for (int j = 0; j < totalInstances; j++) {
				if (data.instance(j).stringValue(attrIndex).equals(currValue)) {
					numCurr++;
					subInstances.add(data.instance(j));
				}
			}
			probX[i] = (double)numCurr/(double)totalInstances;
			entropyYgivenX[i] = numCurr == 0? 0.000000000000: calTotalEntropy(subInstances);
			//System.out.println(entropyYgivenX[i]);
			//System.out.println(currValue);
		}
		double toSubstract = 0.000000000000;
		for (int i = 0; i < numClasses; i++) {
			toSubstract += probX[i]*entropyYgivenX[i];
		}
		
		return totalEntropy - toSubstract;
	}

	private double numericCalcInfoGain(Instances data, int attrIndex, double threshold, double totalEntropy) {
		double[] probX = new double[2];
		double[] entropyYgivenX = new double[2];
		int totalInstances = data.numInstances();
		int left = 0;
		Instances leftSubInstances = new Instances(data);
		Instances rightSubInstances = new Instances(data);
		leftSubInstances.delete();
		rightSubInstances.delete();
		// System.out.println(data.numInstances());
		for (int i = 0; i < totalInstances; i++) {
			if (data.instance(i).value(attrIndex) <= threshold) {
				left++;
				leftSubInstances.add(data.instance(i));
			} else rightSubInstances.add(data.instance(i));
		}
		probX[0] = (double)left/(double)totalInstances;
		probX[1] = (double)1 - probX[0];
		entropyYgivenX[0] = left == 0? 0.000000000000:calTotalEntropy(leftSubInstances);
		entropyYgivenX[1] = left == totalInstances? 0.000000000000:calTotalEntropy(rightSubInstances);
		return totalEntropy - (probX[0]*entropyYgivenX[0] + probX[1]*entropyYgivenX[1]);
	}

}