/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package tutorials.cppn;

import client.ParameterTableInstance;
import client.cppn.InputLayer;
import client.cppn.Neuron;
import java.util.*;

/**
 * Tutorial code that shows how to change the input layer implementation.
 * 
 * @author Nick
 */
public final class ManhattenDistanceInputLayer implements InputLayer {
	private final Neuron x;
	private final Neuron y;
	private final Neuron distance;
	private final Neuron bias;
	
	public ManhattenDistanceInputLayer(Collection <Neuron> neurons) {
		x = find(neurons, "x");
		y = find(neurons, "y");
		distance = find(neurons, "d");
		bias = find(neurons, "bias");
	}
	
	private static Neuron find(Collection<Neuron> neurons, String label) {
		for(Neuron neuron: neurons)
			if(neuron.getNode().getType().equals("in") && neuron.getNode().getLabel().equals(label))
				return neuron;
		return null;
	}
	
	public void writeInput(double x, double y){
		this.x.setInput(x);
		this.y.setInput(y);
		this.distance.setInput(x + y);
		this.bias.setInput(1.0);
	}
}


