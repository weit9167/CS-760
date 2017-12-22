import java.util.*;
import weka.core.*;
import java.io.*;

public class InstanceIndexPair {
	private Instance inst = null;
	private Integer index = -1;

	public InstanceIndexPair(Instance inst, Integer index) {
		this.inst = inst;
		this.index = index;
	}

	public Instance getInstance() {
		return this.inst;
	}

	public Integer getIndex() {
		return this.index;
	}
}