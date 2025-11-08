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

package client;

import java.util.Map;

/**
 * The ParameterTable is the central interface for all server supplied
 * evolution and cppn parameters.
 * 
 * The implementation does not gaurantee efficient retrieval of a parameter,
 * so care should be taken to call these methods only when nescessary.  For
 * example, the implementation may just store an xml document and traverse
 * the document's structure.
 * 
 * @author Nick Beato
 */

public interface ParameterTable {
	/**
	 * Gets a boolean from the parameter table.
	 * 
	 * @param classification The classification of the parameter
	 * @param parameter The name of the parameter in the classification
	 * @return The parameter
	 */
	public boolean getBoolean(String classification, String parameter);
	
	
	/**
	 * Gets a double from the parameter table.
	 * 
	 * @param classification The classification of the parameter
	 * @param parameter The name of the parameter in the classification
	 * @return The parameter
	 */
	public double getDouble(String classification, String parameter);
	
	/**
	 * Gets an integer from the parameter table.
	 * 
	 * @param classification The classification of the parameter
	 * @param parameter The name of the parameter in the classification
	 * @return The parameter
	 */
	public int getInteger(String classification, String parameter);

	/**
	 * Gets a parameter from the parameter table as a String.
	 * 
	 * @param classification The classification of the parameter
	 * @param parameter The name of the parameter in the classification
	 * @return The parameter
	 */
	public String getParameter(String classification, String parameter);
	
	/**
	 * Gets a set of parameters, such as activations, inputs, etc.,
	 * as an array. This method will return null if the set does
	 * not exist.
	 * 
	 * @param setName The name of the set
	 * @return The set of parameters
	 */
	public String []getSetAsArray(String setName);
	
	/**
	 * Gets the weight of the item in the set.
	 * If no such item exists, 0 is returned.
	 * 
	 * @param setName The name of the set
	 * @param item The name of the item
	 * @return The weight of the item
	 */
	public double getWeightOfSetItem(String setName, String item);
	
	/**
	 * Gets a random item from a set. This will return null if the set
	 * does not exist.
	 * 
	 * @param setName The name of the set
	 * @return A random item
	 */
	public String getRandomItemFromSet(String setName);
}


