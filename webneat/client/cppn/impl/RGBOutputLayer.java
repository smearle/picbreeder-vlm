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

import java.util.Collection;

import client.cppn.OutputLayer;
import client.cppn.Neuron;

/**
 * An output layer that interprets color by reading the direct
 * RGB values [0,1] from the red, green, and blue output neurons.
 * 
 * @author Nick
 */

final class RGBOutputLayer implements OutputLayer {
	private final double []output = new double[3];
	private Neuron red, green, blue;
	
	RGBOutputLayer(Collection<Neuron> neurons) {
		for(Neuron n : neurons)
			if(n.getNode().getType().equals("out")) {
				if(n.getNode().getLabel().equals("red")) red = n;
				else if(n.getNode().getLabel().equals("green")) green = n;
				else if(n.getNode().getLabel().equals("blue")) blue = n;
			}
	}
	
	public double []readOutput() {
		output[0] = clamp(red.getOutput());
		output[1] = clamp(green.getOutput());
		output[2] = clamp(blue.getOutput());
		return output;
	}

	private static double clamp(double d) {
		return Math.min(1.0, Math.max(0.0, d));
		//return Math.min(1.0, Math.abs(d));
	}
}
