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

public class XmlParameterTable extends AbstractParameterTable implements Configurable {
	public void configure(Element xmlElement) {
		NodeList children = xmlElement.getChildNodes();
		
		for(int i = 0; i < children.getLength(); i++)
			if(children.item(i) instanceof Element) {
				Element e = (Element) children.item(i);
				
				if(e.getTagName().equals("group"))
					loadGroup(e);
				else if(e.getTagName().equals("set"))
					loadSet(e);
			}
	}
	
	private void loadSet(Element xmlElement) {
		final String setName = xmlElement.getAttribute("name");
		NodeList children = xmlElement.getChildNodes();
		
		for(int i = 0; i < children.getLength(); i++)
			if(children.item(i) instanceof Element) {
				Element e = (Element) children.item(i);
				if(e.hasAttribute("weight"))
					addItemToSet(setName, e.getAttribute("value"), Double.parseDouble(e.getAttribute("weight")));
				else
					addItemToSet(setName, e.getAttribute("value"));
			}
	}
	
	private void loadGroup(Element xmlElement) {
		final String groupName = xmlElement.getAttribute("name");
		NodeList children = xmlElement.getChildNodes();
		
		for(int i = 0; i < children.getLength(); i++)
			if(children.item(i) instanceof Element) {
				Element e = (Element) children.item(i);
				setParameter(groupName, e.getAttribute("name"), e.getAttribute("value"));
			}
	}
}
