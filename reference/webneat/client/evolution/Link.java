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
 * A Link gene connects two node genes (in a directed manner) and
 * models how strong the connection is. This gene maps directly to
 * a CPPN connection. The strength of the connection is the weight
 * in the CPPN, so we'll just call it strength for simplicity.
 * <p>
 * The Link represents the connections as labels, not through
 * explicit object links. The CPPN is responsible for connecting
 * things together beyond a conceptual level.
 * 
 * @author Nick Beato
 */
public interface Link extends Gene {
	/**
	 * Gets the marking of the source node.
	 * 
	 * @return The marking of the source node
	 */
	public Marking getSourceMarking();
	
	/**
	 * Gets the marking of the destination node.
	 * 
	 * @return The marking of the destination node
	 */
	public Marking getDestinationMarking();
	
	/**
	 * Gets the link weight of this gene.
	 * 
	 * @return The link weight
	 */
	public double getWeight();
	
	/**
	 * Sets the link weight of this gene.
	 * <p>
	 * This method should only be invoked by the Generators during evolution.
	 *
	 * @param weight The new weight
	 */
	public void setWeight(double weight);
	
	/**
	 * Queries this link to see if it connects the specified nodes via their
	 * markings.  This is a directed connection.
	 * 
	 * @param source The marking of the source node
	 * @param destination The marking of the destination node
	 * @return <code>true</code> if this connects the markings,
	 * 	<code>false</code> otherwise
	 */
	
	public boolean connects(Marking source, Marking destination);
}


