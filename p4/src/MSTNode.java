import java.util.*;
import weka.core.*;
import java.io.*;

public class MSTNode {
	public ArrayList<String> parents = null;
	//public ArrayList<String> children = null;
	public String nodeName = null; // the name of attribute of this node
	public ArrayList<Double[][]> condProbTableList = null;

	public MSTNode(String attrName, String parent) {
		this.nodeName = attrName;
		this.parents = new ArrayList<String>();
		this.parents.add(parent);
		//this.children = new ArrayList<String>();
		this.condProbTableList = new ArrayList<Double[][]>();
	}
}