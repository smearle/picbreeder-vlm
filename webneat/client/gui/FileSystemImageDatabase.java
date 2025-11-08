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

package client.gui;

import client.ImageDatabase;
import java.awt.Image;
import javax.imageio.ImageIO;

class FileSystemImageDatabase implements ImageDatabase {
	public Image getImage(String name) {
		try {
			return ImageIO.read(getClass().getResource("/" + name));
		}
		catch(Exception e) {
			return null;
		}
	}
}
