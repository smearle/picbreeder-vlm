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

import org.w3c.dom.*;

/**
 * Transferable represents the objects in the system which will be
 * readily transferred to and from the server.  These objects differ
 * from configurable in that they may be modified while stored on the
 * client.  A configurable object cannot do this.
 * 
 * @author Nick Beato
 */
public interface Transferable {
	/**
	 * Loads this object from an xml element.
	 * 
	 * @param xmlElement The xml element 
	 */
	public void load(Element xmlElement);
	
	/**
	 * Saves this object as an xml element.  The provided document should
	 * be used as a factory to create new elements, attributes, and fields.
	 * 
	 * @param xmlElement The element to store in
	 * @param xmlDocument The xml factory object
	 */
	public void store(Element xmlElement, Document xmlDocument);
	
	/**
	 * Gets the name of the xml element associated with this object.
	 * 
	 * @return The xml element name of this object
	 */
	public String getElementName();
}


