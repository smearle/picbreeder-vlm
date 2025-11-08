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

import java.awt.Image;

/**
 * The ImageDatabase is responsible for loading any graphics
 * required by the gui.  This class is nescessary because an
 * application may read directly from the harddrive, but an
 * Applet must read from the server.
 * 
 * @author Nick
 *
 */
public interface ImageDatabase {
	/**
	 * Gets the image with the given relative fileName.
	 * <p>
	 * Because applets behave differently than applications, this method
	 * does not gaurantee any behavior if the file does not exist or
	 * cannot be loaded.
	 * 
	 * @param fileName The relative file name
	 * @return The image for the given file
	 */
	public abstract Image getImage(String fileName);
}
