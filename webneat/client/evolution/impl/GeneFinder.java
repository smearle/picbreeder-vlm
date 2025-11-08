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

package client.evolution.impl;

import java.util.Comparator;

import client.evolution.Gene;
import client.evolution.Marking;

class GeneFinder implements Comparator {
	public int compare(Object gene, Object marking) {
		return ((Gene) gene).getMarking().compareTo((Marking) marking);
	}
}
