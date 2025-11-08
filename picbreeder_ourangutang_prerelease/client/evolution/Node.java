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


package client.evolution;

/**
 * A Node is a gene that represents a neuron in the genome.  
 * It stores an activation function (such as sigmoid, gaussian, etc).
 * It also has attributes to identify the inputs and outputs.
 * 
 * @author Nick Beato
 */
public interface Node extends Gene {
	/**
	 * Gets the activation type used by this node.
	 * 
	 * @return A textual representation of the activation function
	 */
	public String getActivation();
	
	/**
	 * Sets the activation type of this node.
	 * 
	 * @param activation The new activation method
	 */
	public void setActivation(String activation);
	
	/**
	 * Gets the type of this node.  ie, input, hidden, output.
	 * 
	 * @return The type of this node.
	 */
	public String getType();
	
	/**
	 * Gets the label of this node.
	 * 
	 * @return The label of the node, if applicable.
	 */
	public String getLabel();
	
	/**
	 * Queries if this node has a label associated with it.
	 * 
	 * @return <code>true</code> if the label is a valid string,
	 * <code>false</code> otherwise.
	 */
	public boolean hasLabel();
}
