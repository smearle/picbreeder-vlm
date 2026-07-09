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

import java.util.*;

import client.evolution.Link;
import client.evolution.Marking;
import client.evolution.Node;
import client.utilities.*;

/**
 * 
 * @author Nick Beato
 */
class History {

	private Map <Marking, Marking> linkMarkings;
	private Map <Pair <Marking, Marking>, Marking> nodeMarkings;
	
	History() {
		linkMarkings = new HashMap <Marking, Marking> ();
		nodeMarkings = new HashMap <Pair <Marking, Marking>, Marking> ();
	}
	
	/**
	 * <p>Does ...</p>
	 * 
	 */
	public void clear() {
		linkMarkings.clear();
		nodeMarkings.clear();
	}
	
	public Marking findNodeMarkingFromLink(Link parent) {
		return linkMarkings.get(parent.getMarking());
	}
	
	public Marking findLinkMarkingFromNodes(Node from, Node to) {
		return nodeMarkings.get(new Pair <Marking, Marking> (from.getMarking(), to.getMarking()));
	}
	
	public void updateNodeMarkingFromLink(Link parent, Marking marking) {
		linkMarkings.put(parent.getMarking(), marking);
	}
	
	public void updateLinkMarkingFromNodes(Node from, Node to, Marking marking) {
		nodeMarkings.put(new Pair <Marking, Marking> (from.getMarking(), to.getMarking()), marking);
	}
}
