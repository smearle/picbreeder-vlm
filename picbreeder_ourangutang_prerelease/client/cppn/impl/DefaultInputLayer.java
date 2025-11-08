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

package client.cppn.impl;

import client.ParameterTableInstance;
import client.cppn.InputLayer;
import client.cppn.Neuron;
import java.util.*;

/**
 * 
 * @author Adam Campbell
 */
final class DefaultInputLayer implements InputLayer {
	private final double DISTANCE_SCALE;
	private final double X_SCALE;
	private final double Y_SCALE;
	private final double BIAS;
	
	/**
	 * Represents the neuron containing the x input value.
	 *
	 */
	private DefaultNeuron xInput;
	/**
	 * Represents the neuron containing the y input value.
	 *
	 */
	private DefaultNeuron yInput;
	/**
	 * Represents the neuron containing the distance from origin input value.
	 *
	 */
	private DefaultNeuron distInput;

	/**
	 * Represents the neuron containing the bias value.
	 *
	 */
	private DefaultNeuron biasInput;
	
	/**
	 * Constructs an input layer out of the list of neurons given.
	 * The inputs are x, y, and distance from the origin.
	 *
	 * @param neurons List of neurons from which this input layer
	 * 	will obtain the correct input nodes.
	 */
	public DefaultInputLayer(Collection <Neuron> neurons){
		// statics cause problems in applet :(
		BIAS = ParameterTableInstance.get().getDouble("activation", "bias");
		X_SCALE = ParameterTableInstance.get().getDouble("activation", "x scale");
		Y_SCALE = ParameterTableInstance.get().getDouble("activation", "y scale");
		DISTANCE_SCALE = ParameterTableInstance.get().getDouble("activation", "distance scale");
		
		xInput = find(neurons, "x");
		yInput = find(neurons, "y");
		distInput = find(neurons, "d");
		biasInput = find(neurons, "bias");
	}
	
	/**
	 * Returns an input neuron with a specific label.  If no neuron in the
	 * collection matches the criteria, then null is returned.
	 *
	 * @param neurons Collection of neurons to be searched.
	 * @param label Label of the sought after neuron.
	 * @return Input neuron with given label.  If none is found, then null is returned.
	 */
	private static DefaultNeuron find(Collection<Neuron> neurons, String label){
		for(Neuron neuron: neurons){
			if(neuron.getNode().getType().equals("in") && neuron.getNode().getLabel().equals(label)){
				return (DefaultNeuron)neuron;
			}
		}
		return null;
	}
	
	public void writeInput(double x, double y){
		x *= X_SCALE;
		y *= Y_SCALE;
		
		xInput.setInput(x);
		yInput.setInput(y);
		distInput.setInput(distance(x, y) * DISTANCE_SCALE);
		biasInput.setInput(BIAS);
	}
	
	/**
	 * Returns the distance of the given (x,y) pair to the origin.
	 *
	 * @param x Value in the x direction.
	 * @param y Value in the y direction.
	 * @return Distance of the x,y pair to the origin.
	 */
	private static double distance(double x, double y){
		return Math.sqrt(x*x + y*y);
	}
	
}


