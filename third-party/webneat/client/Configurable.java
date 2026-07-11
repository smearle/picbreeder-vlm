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
 * Configurable represents the objects in the system that can be configured
 * during initialization.  These objects must be able to parse their
 * data from an XML DOM document.   The same document will be used to store
 * all configuration information, so each type of object must be stored in
 * a second level element named after the class. 
 * 
 * @author Nick Beato
 */

public interface Configurable {
	/**
	 * Configures this object from an xml element.
	 * 
	 * @param xmlElement The xml element representing this object 
	 */
	public void configure(Element xmlElement);
}
