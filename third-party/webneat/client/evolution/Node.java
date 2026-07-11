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
	
	/**
	 * Sets the label. Useful in mutation schemes that might convert
	 * genomes.
	 * 
	 * @param label The new label
	 */
	public void setLabel(String label);

	/**
	 * Gets the affinity of this gene. The affinity is an abstract concept
	 * that groups genes together during evolution.
	 * <p>
	 * For example, a network that wants to preprocess the x input could
	 * set the affinity of the x neuron to a unique value that forces an
	 * x "subnet" to evolve more tightly then the remaining network.
	 * <p>
	 * This was added for the use of color subnets, but is much more general.
	 * The color subnet is actually part of the same genome, but the affinity
	 * forces the color subnet to occur "later" in activation and prevents
	 * the color subnet from "feeding into" the "ink" network that produces
	 * the greyscale values.
	 * 
	 * @return The affinity of this gene
	 */
	public String getAffinity();
	
	/**
	 * Sets the affinity of this gene.
	 * <p>
	 * This is a utility for gene conversion. I'd suggest not using it
	 * and letting the factories do this for you.
	 * @param affinity
	 */
	public void setAffinity(String affinity);
	
	/**
	 * Gets the bias of this node.
	 * 
	 * @return The bias
	 */
	public double getBias();

	/**
	 * Sets the bias of this node.
	 * 
	 * @param bias The bias
	 */
	public void setBias(double bias);
}
