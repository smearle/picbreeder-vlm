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

import client.evolution.Series;

public class SeriesInstance {
	private static Series singleton = null;
	
	public static Series get() {
		return singleton;
	}
	
	public static void set(Series instance) {
		singleton = instance;
	}
}
