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

import client.evolution.Marking;

final class DefaultMarking extends BasicInfo implements Marking {
	public DefaultMarking(long number) {
		super(number);
	}
	
	public int compareTo(Marking other) {
		if(usesCurrentBranch()) {
			if(other.usesCurrentBranch())
				return (int)(id - other.getId());
			else
				return -1;
		}
		else if(other.usesCurrentBranch())
			return 1;
		
		int t = getBranch().compareTo(other.getBranch());
		if(t != 0)
			return t;
		else
			return (int)(id - other.getId());
	}
	
	public String getElementName() {
		return "marking";
	}
}
