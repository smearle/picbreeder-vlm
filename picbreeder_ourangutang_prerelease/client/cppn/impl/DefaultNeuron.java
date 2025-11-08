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
import client.cppn.Neuron;
import client.evolution.*;
import client.math.*;
/**
 * 
 * @author Adam Campbell
 */
final class DefaultNeuron implements Neuron {
	/**
	 * Represents the activation function for this neuron.
	 *
	 */
	final private Function function;
	
	/**
	 * Represents the input values obtained from its incoming connections.
	 *
	 */
	private double input;
	
	/**
	 * Represents the value output from this neuron.
	 *
	 */
	private double output;
	
	/**
	 * Used to keep values between activations.
	 *
	 */
	private double resetValue;
	
	/**
	 * Represents the node that this neuron was created from.
	 *
	 */
	final private Node node;
	
	/**
	 * Shows whether or not the neuron is active.  If a neuron is active,
	 * then its  output value can be read by the connections using it as
	 * their source.
	 *
	 */
	private boolean active;
	
	/**
	 * Constructs a neuron class given its corresponding evolutionary neuron.
	 * This neuron obtains its activation function from the evolutionary neuron.
	 *
	 * @param neuron The evolutionary neuron from which this neuron obtains its
	 * 	activation function.
	 */
	public DefaultNeuron(Node n){
		node = n;
		function = FunctionParser.instance().parse(n.getActivation());
		resetValue = 0.0;
		active = false;
	}
	
	public void activate(){
		if(active){
			output = function.valueAt(input);
			input = resetValue;
		}
	}
	
	public void setInput(double x){
		input = resetValue = x;
		//Set this neuron to be active because its input value was explicitly set.
		//A neuron's input is explicitly set only if it is an input neuron, and input neurons
		//are always active.
		active = true;
	}
	
	public Node getNode(){
		return node;
	}
	
	public void addInput(double x){
		input += x;
	}
	
	public void clearInput(){
		input = resetValue = 0.0;
		active = false;
	}
	
	public double getOutput(){
		return output;
	}

	public boolean isActive(){
		return active;
	}
	
	public void setActive(boolean b){
		active = b;
	}
}


