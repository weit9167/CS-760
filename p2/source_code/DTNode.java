import weka.core.*;
import java.util.*;

public class DTNode {
	// fields
	private boolean isLeaf;
	public Attribute attribute;
	public String attrValue;
	public double threshold;
	public int side;
	public String label;
	public ArrayList<DTNode> children;
	public ArrayList<ArrayList<Integer>> numInstances; // the numbers in square braket

	public DTNode (boolean isLeaf, Attribute attribute, String attrValue, double threshold, int side, String label) {
		this.isLeaf = isLeaf;
		this.attribute = attribute;
		this.label = label;
		this.children = new ArrayList<DTNode>();
		this.numInstances = new ArrayList<ArrayList<Integer>>(); 
		this.attrValue = attrValue;
		this.threshold = threshold;
		this.side = side;
	}

	public boolean isLeaf() {
		return isLeaf;
	}
}